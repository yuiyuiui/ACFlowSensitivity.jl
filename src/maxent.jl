# 一些说明：
# 1. 为了方便计算，我们统一使用均匀的输出网格点，并记个点间距为d
Zygote.@nograd range

# maximum entropy context
struct MEContext
	d::Float64   # output gap
	output_weight::Vector{Float64}
	K::Matrix{Float64}
	U::Matrix{Float64}
	V::Matrix{Float64}
	S::Vector{Float64}
	n::Int64    # dimension of singular space 
	model::Vector{Float64}
	α_vec::Vector{Float64}
	E::Matrix{Float64}    # unit matrix
end


#= wirte optimize max Q by hand for AD
function my_chi2kink(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
	output_number = length(output_range)
	N = length(Gvalue)

	# 计算积分时候网格点的权重
	d = output_range[2] - output_range[1]
	output_weight = fill(d, output_number)

	# set the kernel matrix
	kernel = Matrix{ComplexF64}(undef, N, output_number)
	for i ∈ 1:N
		for j ∈ 1:output_number
			kernel[i, j] = 1 / (iwn[i] - output_range[j])
		end
	end

	# real paraliaze Gvalue and kernel
	G = vcat(real(Gvalue), imag(Gvalue))
	K = [real(kernel); imag(kernel)]
	U, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]
	U = U[:, 1:n]
	S = S[1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 4)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 16
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end
	χ²_vec = Vector{Float64}(undef, L)

	# 后面log10(α)和log10(χ²)要拟合的曲线
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	# 拟合曲线时候为了防止过拟合设置的参数
	#adjust = 2.5


	# function Q
	A_vec(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
	Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

	# -𝞉Q/∂A
	J(u::Vector{Float64}, α::Float64) = α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

	# -∂²Q/∂A∂u
	H(u::Vector{Float64}, α::Float64) = α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V



	# 接下来用Newton method求最值点
	u_guess=zeros(n)
	for i in 1:L
		@show i
		α = α_vec[i]
		u_opt, call = my_newton(u -> J(u, α), u -> H(u, α), u_guess)
		u_guess = copy(u_opt)
		χ²_vec[i] = χ²(u_opt)
		@show log10(α),log10(χ²_vec[i]),norm(J(u_opt,α)),call
	end
	idx = findall(isfinite,χ²_vec)
	α_vec=α_vec[idx]
	χ²_vec=χ²_vec[idx]

	# 现在进行曲线拟合
	guess_fit = [0.0, 5.0, 2.0, 0.0]
	_, _, c, dd = my_curve_fit(log10.(α_vec), log10.(χ²_vec), guess_fit)


	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c-2.5/dd)
	u_opt,_ = my_newton(u -> J(u, α_opt), u -> H(u, α_opt), zeros(n))

	#复原返回要求的A
	return A_vec(u_opt)
end
=#

function AD_chi2kink(iwn::Vector{ComplexF64}, G::Vector{Float64}, output_range::Vector{Float64})
	mec = MEContext_compute(iwn, output_range)
	return Zygote.jacobian(x->chi2kinkForAD(x,mec),G)
end


function chi2kinkForAD( G::Vector{Float64}, mec::MEContext)
	d = mec.d
    output_weight = mec.output_weight
    K = mec.K
    U = mec.U
    V = mec.V
    S = mec.S
	n = mec.n
    model = mec.model
    α_vec = mec.α_vec
    E = mec.E

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 16
	χ²_vec = zeros(L)

	# 后面log10(α)和log10(χ²)要拟合的曲线
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	# function Q
	A_vec(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
	Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

	# -𝞉Q/∂A
	J(u::Vector{Float64}, α::Float64) = α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

	# -∂²Q/∂A∂u
	H(u::Vector{Float64}, α::Float64) = α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V

	# 接下来用Newton method求最值点
	u_guess=zeros(n)
	for i in 1:L
		@show i
		α = α_vec[i]
		u_opt, call = my_newton(u -> J(u, α), u -> H(u, α), u_guess)
		u_guess = copy(u_opt)
		χ²_vec += χ²(u_opt)*E[:,i]
		@show log10(α),log10(χ²_vec[i]),norm(J(u_opt,α)),call
	end
	idx = findall(isfinite,χ²_vec)
	α_vec0=α_vec[idx]
	χ²_vec0=χ²_vec[idx]

	# 现在进行曲线拟合
	guess_fit = [0.0, 5.0, 2.0, 0.0]
	_, _, c, dd = my_curve_fit(log10.(α_vec0), log10.(χ²_vec0), guess_fit)


	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c-2.5/dd)
	u_opt,_ = my_newton(u -> J(u, α_opt), u -> H(u, α_opt), zeros(n))

	#复原返回要求的A
	return A_vec(u_opt)
end

function MEContext_compute(iwn::Vector{ComplexF64},  output_range::Vector{Float64})
	output_number = length(output_range)
	N = length(iwn)

	# 计算积分时候网格点的权重
	d = output_range[2] - output_range[1]
	output_weight = fill(d, output_number)

	# set the kernel matrix
	kernel = Matrix{ComplexF64}(undef, N, output_number)
	for i ∈ 1:N
		for j ∈ 1:output_number
			kernel[i, j] = 1 / (iwn[i] - output_range[j])
		end
	end

	# real paraliaze kernel
	K = [real(kernel); imag(kernel)]
	U, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]
	U = U[:, 1:n]
	S = S[1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 4)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	L = 16
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end

	# For AD on mutable vector χ²_vec, set a series of eᵢ
	E = Matrix{Float64}(I(L))

	return MEContext(d,output_weight,K,U,V,S,n,model,α_vec,E)
end















