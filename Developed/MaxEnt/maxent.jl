mutable struct MaxEntContext
    Gᵥ::Vector{F64}
    σ²::Vector{F64}
    grid::AbstractGrid
    mesh::AbstractMesh
    model::Vector{F64}
    kernel::Array{F64,2}
    hess::Array{F64,2}
    Vₛ::Array{F64,2}
    W₂::Array{F64,2}
    W₃::Array{F64,3}
    Bₘ::Vector{F64}
end


"""
	TangentMesh

Mutable struct. A non-linear and non-uniform mesh. Note that it should
be defined on both negative and positive half-axis.

### Members
* nmesh  -> Number of mesh points.
* wmax   -> Right boundary (maximum value).
* wmin   -> Left boundary (minimum value).
* mesh   -> Mesh itself.
* weight -> Precomputed integration weights (composite trapezoidal rule).

See also: [`LinearMesh`](@ref).
"""
mutable struct TangentMesh{T} <: AbstractMesh
    nmesh::I64
    wmax::T
    wmin::T
    mesh::Vector{T}
    weight::Vector{T}
end




"""
    RawData

Mutable struct. It represent the raw input data. The datatype `T` of raw
data may be `Float64` or `ComplexF64`.

### Members
* _grid -> Raw grid for the input data, such as τ or iωₙ.
* value -> Raw input data, such as G(τ), G(iωₙ), or Σ(iωₙ).
* error -> Error bar (standard deviation) of raw input data, σ.

See also: [`GreenData`](@ref).
"""
mutable struct RawData{T} <: AbstractData
    _grid::Vector{F64}
    value::Vector{T}
    error::Vector{T}
end






# ————————————————————————————————————————————————————————————————————————
# 一层函数

function solve(S::MaxEntSolver, rd::RawData)
    println("[ MaxEnt ]")
    #
    mec = init(S, rd)
    darr, sol = run(mec)
    gout = last(mec, darr, sol)
    #
    return mec.mesh.mesh, sol[:A], gout
end

function init(S::MaxEntSolver, rd::RawData)
    # Prepera input data
    G = make_data(rd)
    Gᵥ = G.value
    #较大的误差会导致不稳定的结果，因此给予更小的权重
    σ² = 1.0 ./ G.covar

    # Prepare grid for input data
    grid = make_grid(rd)

    # Prepare mesh for output spectrum
    mesh = make_mesh()

    # Prepare default model function
    # 这是一个长度为mesh，积分为1，限制在mesh上的正态函数
    model = make_model(mesh)

    # Prepare kernel function
    kernel = make_kernel(mesh, grid)

    # Prepare some essential intermediate variables
    Vₛ, W₂, W₃, Bₘ, hess = precompute(Gᵥ, σ², mesh, model, kenel)

    return MaxEntContext(Gᵥ, σ², grid, mesh, model, kernel, hess, Vₛ, W₂, W₃, Bₘ)
end


function run(mec::MaxEntContext)
    method = get_m("method")
    @cswitch method begin
        @case "historic"
        return historic(mec)
        break

        @case "classic"
        return classic(mec)
        break

        @case "bryan"
        return bryan(mec)
        break

        @case "chi2kink"
        return chi2kink(mec)
        break
    end
end

function last() end

# ————————————————————————————————————————————————————————————————————————
# 二层函数

function precompute(
    Gᵥ::Vector{F64},
    σ²::Vector{F64},
    am::AbstractMesh,
    model::Vector{F64},
    kernel::Matrix{F64},
)

    # Create singular value space
    U, V, S = make_singular_space(kernel)

    # Evaluate sizes of the arrays
    nmesh = length(am)
    n_svd = length(S)

    # Allocate memories
    W₂ = zeros(F64, n_svd, nmesh)
    W₃ = zeros(F64, n_svd, n_svd, nmesh)
    Bₘ = zeros(F64, n_svd)
    hess = zeros(F64, nmesh, nmesh)

    # Get weight of the mesh, Δωₗ.
    mesh_weight = am.weight

    # Compute Wₘₗ
    @einsum W₂[m, l] =
        σ²[k] * (U[k, n] * U[k, m]) * (S[m] * S[n] * V[l, n]) * (mesh_weight[l] * model[l])

    # Compute Wₘₗᵢ
    @einsum W₃[m, k, l] = W₂[m, l] * V[l, k]

    # Compute Bₘ
    @einsum Bₘ[m] = S[m] * U[k, m] * σ²[k] * Gᵥ[k]

    # Compute the Hessian matrix
    @einsum hess[i, j] =
        mesh_weight[i] * mesh_weight[j] * (kernel[k, i] * kernel[k, j] * σ²[k])

    return V, W₂, W₃, Bₘ, hess
end

function chi2kink(mec::MaxEntComtext)

    # a function to 拟合 log(χ²) and log(α)
    function fitfun(x, p)
        return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
    end

    println("Apply chi2kink algorithm to determine optimized α")

    use_bayes = false
    α = get_m("alpha")
    ratio = get_m("ratio")
    nalpha = get_m("nalpha")

    #最终的最小的α
    α_end = α / (ratio^nalpha)

    n_svd = length(mec.Bₘ)

    # medium value of the optimazation process
    sol_now = zeros(F64, n_svd)

    # Store the series of solutions
    sol_vec = []

    χ²_vec = []
    α_vec = []

    while true
        sol = optimizer(mec, α, slo_now, use_bayes)
        push!(sol_vec, sol)
        push!(α_vec, alpha)
        push!(χ²_vec, sol[:χ²])

        # Update iteration value
        @. sol_now = sol[:u]
        alpha = alpha / ratio
        if alpha < α_end
            break
        end
    end

    # 有意义的χ²
    good = isfinite.(χ²_vec)

    # 拟合曲线的四个参数
    guess = [0.0, 5.0, 2.0, 0.0]

    # 开始拟合，这里guess代表接下来要用的拟合方法的初始值
    fit = curve_fit(fitfun, log10.(α_vec[good]), log10.(χ²_vec[good]), guess)
    _, _, c, d = fit.param

    # 防止过拟合或者欠拟合
    #原理在于，以d>0时候为例，对f(x)=1/(1+exp(-dx)), f(x) increases slowly when x<0 and fast when 
    # x>0. So if we get a c>c0 which is the real value, get c'=c-1/2 f'(c)=c-2/d can 
    # reduce the error effectively. And if we get c≃c0, because f(x) change slowly when 
    # x<c, do c'=c-2/d won't make too much error.
    fit_pos = 2.5

    α_opt = c - fit_pos / d

    # 选取距离“拐点“最近α
    α_chosen = argmin(abs.(log10(α_vec) .- α_opt))

    sol_now = sol_vec[α_chosen][:u]

    # trans log10(α)t to origin scale
    α_opt = 10.0^α_opt

    sol = optimizer(mec, α_opt, slo_now, use_bayes)

    return slo_vec, sol
end


# ————————————————————————————————————————————————————————————————————————
# 三层函数


# For given α, calculate corresponding
# 
function optimizer(
    mec::MaxEntContext,
    α::Float64,
    sol_now::Vector{Float64},
    use_bayes::Bool,
)

    blur = get_("blur")
    offdiag = get_b("offdiag")

    if offdiag
        solution, call = newton(f_and_J_od, sol_now, mec, α)
        sol = copy(solution)
        A = svd_to_real_od(mec, solution)
        S = calc_entropy_od(mec, A)
    else
        solution, call = newton(f_and_J, sol_now, mec, α)
        sol = copy(solution)
        A = svd_to_real(mec, solution)
        S = calc_entropy(mec, A, sol)
    end

    χ² = calc_chi2(mec, A)
    # 计算数值积分
    norm = trapz(mec, mesh, A)

    dict = Dict{Symbol,Any}(
        :sol => sol,
        :α => α,
        :S => S,
        :χ² => χ²,
        :norm => norm,
        :Q => α * S - 0.5 * χ²,
        :Araw => deepcopy(A),
    )

    if use_bayes
        if offdiag
            ng, tr, conv, prob = calc_bayes_od(mec, A, S, χ², α)
        else
            ng, tr, conv, prob = calc_bayes(mec, A, S, χ², α)
        end

        # ng：表示有效数据点的数量。
        # tr：跟踪矩阵的迹。
        # conv：收敛情况。
        # prob：贝叶斯概率。
        dict[:ngood] = ng
        dict[:trace] = tr
        dict[:conv] = conv
        dict[:prob] = prob
    end

    # 模糊化参数 blur
    if blur > 0.0
        make_blur(mec.mesh, A, blur)
    end
    dict[:A] = A

    return dict
end

# model: dit function
# x: log10(α)
# y: loh10(χ²)
# p0: origin solution of iteration
function curve_fit(model, x::AbstractArray, y::AbstractArray, p0::AbstractArray)
    f = p -> model(x, p) - y
    r = f(p0)
    # construct a differential objection 
    R = OnceDiffrerntial(f, p0, r)
    OR = levenberg_marquardt(R, p0)
    p = OR.minimizer
    # 算法的收敛性
    conv = OR.xconv || OR.gconv
    return LsqFitResult(p, value!(R, p), jacobian!(R, p), conv)
end


# ————————————————————————————————————————————————————————————————————————
# 四层函数


# 根据当前解（feed）、目标函数值（f）和雅可比矩阵（J）
function newton(
    fun::Function,
    guess,
    kwargs...;
    maxiter::Int64 = 20000,
    mixing::Float64 = 0.5,
)

    # 前面加下划线表明这是不希望被外部调用的内部函数
    # feed s是当前猜测值
    function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = 1.0
        limit = 1e-4
        try
            # 关于pinv函数，pinv函数指的是伪逆，也就是摩尔-彭若斯伪逆（Moore-Penrose Pseudoinverse）
            # 记A的伪逆是A⁺，则A⁺满足
            # (1) AA⁺A=A
            # (2) A⁺AA⁺=A⁺
            # (3) (AA⁺)ᴴ=AA⁺
            # (4) (A⁺A)ᴴ=A⁺A
            # 满足这四个条件的A⁺是存在且唯一的，且是最小二乘||Ax-b||的解 x=A⁺b
            # 计算方法：A=USVᴴ, then A⁺=VS⁺Uᴴ               
            resid = -pinv(J) * f
        catch
            resid = zeros(Float64, length(feed))
        end

        if any(x -> x > limit, abs.(feed))
            ratio = abs.(resid ./ feed)
            max_ratio = maximum(ratio[abs.(feed) .> limit])
            if max_ratio > 1.0
                step = 1.0 / max_ratio
            end
        end
        return feed + step .* resid
    end
    counter = 0
    feeds = []
    backs = []
    f, J = fun(guess, kwargs...)
    back = _apply(guess, f, J)
    push!(feeds, guess)
    push!(backs, back)

    while true
        counter += 1
        feed = feeds[end] + mixing * (backs[end] - feeds[end])

        f, J = fun(feed, kwargs...)
        back = _apply(feed, f, J)
        push!(feeds, feed)
        push!(backs, back)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum(abs.(back - feed)) < 1e-4
            break
        end
    end

    counter > maxiter && error("Tolerance is reached in newton()!")

    return back, counter
end

function svd_to_real_od() end

function calc_entropy_od() end

function svd_to_real(mec::MaxEntContext, u::Vector{F64})
    stype = get_m("stype")
    if stype == "sj"
        w=exp.(mec.Vₛ*u)
        return mec.model .* w
    else
        w=mec.Vₛ*u
        return mec.model ./ (1.0-mec.model .* w)
    end
end

function calc_entropy() end

function calc_chi2() end



function calc_bayes_od() end

function calc_bayes() end


function OnceDifferentiable() end

function levenberg_marquardt() end

# 五层函数
function f_and_J_od() end

function f_and_J() end


















using Einsum

A = [1 0; 0 2] .+ 0.0
B = [1 2; 3 4] .+ 0.0
C = zeros(2, 2)
@einsum C[i, j] = A[i, k] * B[k, j]


#-------------------------\
# check functions
# check newton

using LinearAlgebra, Zygote

#=
function newton(fun::Function,
	guess,
	kwargs...;
	maxiter::Int64 = 2000,
	mixing::Float64 = 0.5,
 )

	# 前面加下划线表明这是不希望被外部调用的内部函数
	# feed s是当前猜测值
	function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where T
		resid = nothing
		step = 1.0
		limit = 1e-4
		try
			# 关于pinv函数，pinv函数指的是伪逆，也就是摩尔-彭若斯伪逆（Moore-Penrose Pseudoinverse）
			# 记A的伪逆是A⁺，则A⁺满足
			# (1) AA⁺A=A
			# (2) A⁺AA⁺=A⁺
			# (3) (AA⁺)ᴴ=AA⁺
			# (4) (A⁺A)ᴴ=A⁺A
			# 满足这四个条件的A⁺是存在且唯一的，且是最小二乘||Ax-b||的解 x=A⁺b
			# 计算方法：A=USVᴴ, then A⁺=VS⁺Uᴴ               
			resid = -pinv(J) * f
		catch
			resid = zeros(Float64, length(feed))
		end

		if any(x -> x > limit, abs.(feed))
			ratio = abs.(resid ./ feed)
			max_ratio = maximum(ratio[abs.(feed).>limit])
			if max_ratio > 1.0
				step = 1.0 / max_ratio
			end
		end
		return feed + step .* resid
	end

	counter = 0
    # 预分配空间，避免动态改变数组大小
    max_size = maxiter + 1  # 预留空间
    feeds = Vector{Vector{Float64}}(undef, max_size)
    backs = Vector{Vector{Float64}}(undef, max_size)

	f, J = fun(guess, kwargs...)
	feeds[1] = guess  # 初始值
    backs[1] = _apply(guess,f,J)

	while true
		counter += 1
		feed = feeds[counter] + mixing * (backs[counter] - feeds[counter])

        # 这里的J就是正常的数学上的jacobian matrix (𝞉fᵢ/∂xⱼ)ᵢⱼ
		f, J = fun(feed, kwargs...)
		backs[counter+1]=_apply(feed,f,J)
        feeds[counter+1]=feed

		any(isnan.(backs[counter + 1])) && error("Got NaN!")

		if counter >= maxiter || maximum(abs.(backs[counter+1] - feed)) < 1e-4
			break
		end
	end

	counter >= maxiter && error("Tolerance is reached in newton()!")

	return backs[counter+1]
end
=#

function newton(
    fun::Function,
    guess,
    kwargs...;
    maxiter::Int64 = 2000,
    mixing::Float64 = 0.5,
)

    # 前面加下划线表明这是不希望被外部调用的内部函数
    # feed s是当前猜测值
    function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = 1.0
        limit = 1e-4
        try
            # 关于pinv函数，pinv函数指的是伪逆，也就是摩尔-彭若斯伪逆（Moore-Penrose Pseudoinverse）
            # 记A的伪逆是A⁺，则A⁺满足
            # (1) AA⁺A=A
            # (2) A⁺AA⁺=A⁺
            # (3) (AA⁺)ᴴ=AA⁺
            # (4) (A⁺A)ᴴ=A⁺A
            # 满足这四个条件的A⁺是存在且唯一的，且是最小二乘||Ax-b||的解 x=A⁺b
            # 计算方法：A=USVᴴ, then A⁺=VS⁺Uᴴ               
            resid = -pinv(J) * f
        catch
            resid = zeros(Float64, length(feed))
        end

        if any(x -> x > limit, abs.(feed))
            ratio = abs.(resid ./ feed)
            max_ratio = maximum(ratio[abs.(feed) .> limit])
            if max_ratio > 1.0
                step = 1.0 / max_ratio
            end
        end
        return feed + step .* resid
    end

    counter = 0
    # 预分配空间，避免动态改变数组大小
    max_size = maxiter + 1  # 预留空间
    N=length(guess)

    feeds = Vector{Vector{Float64}}(undef, max_size)
    backs = Vector{Vector{Float64}}(undef, max_size)

    f, J = fun(guess, kwargs...)
    feeds[1] = guess  # 初始值
    backs[1] = _apply(guess, f, J)

    while true
        counter += 1
        feed = feeds[counter] + mixing * (backs[counter] - feeds[counter])

        # 这里的J就是正常的数学上的jacobian matrix (𝞉fᵢ/∂xⱼ)ᵢⱼ
        f, J = fun(feed, kwargs...)
        backs[counter+1]=_apply(feed, f, J)
        feeds[counter+1]=feed

        any(isnan.(backs[counter+1])) && error("Got NaN!")

        if counter >= maxiter || maximum(abs.(backs[counter+1] - feed)) < 1e-4
            break
        end
    end


    return backs[counter+1]
end

my_fun(x)=[(x[1]-1.0)^2+(x[2]-3.0)^4], [2.0*(x[1]-1.0) 4.0*(x[2]-3.0)^3]
guess=[0.0, 0.0]

newton(my_fun, guess)

jacobian(x->newton(my_fun, x), [0.0, 0.0])[1]


function my_func1(A::Matrix{Float64})
    A=A .+ 1.0
    return A
end
A=rand(2, 2)
my_func1(A)
jacobian(my_func1, A)
