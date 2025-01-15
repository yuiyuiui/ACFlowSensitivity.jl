# we directly use ∂Q/∂u and ∂²Q/∂u²


using ACFlowSensitivity, LinearAlgebra, Plots
μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 401;
noise = 0.0;
Gvalue = generate_G_values_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im;



# function of chi2kink
output_number = length(output_range);

# 计算积分时候网格点的权重
d = output_range[2] - output_range[1];
output_weight = fill(d, output_number);

# set the kernel matrix
kernel = Matrix{ComplexF64}(undef, N, output_number);
for i ∈ 1:N
	for j ∈ 1:output_number
		kernel[i, j] = 1 / (iwn[i] - output_range[j])
	end
end;

# real paraliaze Gvalue and kernel
G = vcat(real(Gvalue), imag(Gvalue));
K = [real(kernel); imag(kernel)];
_, S, V = svd(K);
n = count(x -> (x >= 1e-10), S);
V = V[:, 1:n];


# defualt model
model = exp.(-output_range .^ 2 / 2);
# 调整参数，归一化
model = model / (model' * output_weight);

# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
σ = 1e-4;

α = 1e4;


# function Q
A_vec(u::Vector{Float64}) = model .* exp.(V * u)
χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

# 𝞉Q/∂u
function ∂Qdiv∂u(u::Vector{Float64}, α::Float64)
    ∂Sdiv∂A = -d * (V * u)'    #行向量	
    ∂χ²div∂A = 2 / (σ^2) * (-d * G' * K + d^2 * A_vec(u)' * K' * K)    #行向量
    ∂Adiv∂u = diagm(A_vec(u)) * V
    ∂Sdiv∂u = ∂Sdiv∂A * ∂Adiv∂u
    ∂χ²div∂u = ∂χ²div∂A * ∂Adiv∂u

    return (α * ∂Sdiv∂u - ∂χ²div∂u / 2)'
end

# Hessel Matrix
function hessel(u::Vector{Float64},α::Float64)
    Av=A_vec(u)
    Am=diagm(Av)
    ∂²S=-d*V'*diagm(Av.*log.(Av./model)+Av)*V
    ∂²χ²=2*d/(σ^2)*V'*( diagm( (-G'*K+d*Av'*K'*K)' ) + d*Am*K'*K )*Am*V
    return α*∂²S-∂²χ²/2
end

#=
function my_newton(J,H,guess::Vector{Float64};tol=1e-6,max_iter=100)
    ite=0
    res=copy(guess)
    while true
        if ite>=max_iter || norm(J(res))<tol
            println("Iteration is $ite")
            return res
        end
        @show res
        @show H(res)
        res -= pinv(H(res))*J(res)
        ite+=1
    end
end
=#

function newton(
    fun::Function,
    grad::Function,
    guess;
    maxiter::Int64 = 20000,
    mixing::Float64 = 0.5
    )
    function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = 1.0
        limit = 1e-4
        try
            resid = - pinv(J) * f
        catch
            resid = zeros(Float64, length(feed))
        end
        if any(x -> x > limit, abs.(feed))
            ratio = abs.(resid ./ feed)
            max_ratio = maximum( ratio[ abs.(feed) .> limit ] )
            if max_ratio > 1.0
                step = 1.0 / max_ratio
            end
        end
        return feed + step .* resid
    end

    counter = 0
    feeds = []
    backs = []

    f = fun(guess)
    J = grad(guess)
    back = _apply(guess, f, J)
    push!(feeds, guess)
    push!(backs, back)
    res_feed=nothing

    while true
        counter = counter + 1
        feed = feeds[end] + mixing * (backs[end] - feeds[end])

        f = fun(feed)
        J = grad(feed)
        back = _apply(feed, f, J)
        push!(feeds, feed)
        push!(backs, back)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum( abs.(back - feed) ) < 1.e-4
            res_feed=feed
            break
        end
    end

    @show norm(back),norm(res_feed),maximum( abs.(back - res_feed) ) < 1.e-4

    counter > maxiter && error("Tolerance is reached in newton()!")

    return back, counter
end

u_opt,iter_num=newton(u->-∂Qdiv∂u(u,α),u->-hessel(u,α),zeros(n))
norm(∂Qdiv∂u(u_opt,α))

norm(∂Qdiv∂u(u_opt,α))





# 检查梯度和hessel矩阵公式的正确性


u=rand(n)
e=1e-6
fidi=zeros(n)
for i=1:n
    u1=copy(u)
    u1[i]+=e
    fidi[i]=(Q(u1,α)-Q(u,α))/e
end

form=∂Qdiv∂u(u,α)
norm(form-fidi)/norm(fidi)


u=rand(n)
e=1e-4
u1=copy(u)
u1[3]+=e
v1=(∂Qdiv∂u(u1,α)-∂Qdiv∂u(u,α))/e
v2=hessel(u,α)[:,3]
norm(v1)
norm(v1-v2)/norm(v1)



# 检查牛顿法本身的正确性
using LinearAlgebra
u=rand(40)
my(x) = sum((x .- 1.1).^4)
grad(x) = 4*(x.-1.1).^3
hessel(x) = diagm(12*(x .- 1.1).^2)
sol=newton(grad,hessel,zeros(40))
norm(grad(sol))






