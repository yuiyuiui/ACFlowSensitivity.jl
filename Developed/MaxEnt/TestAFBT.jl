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
Gvalue = generate_GFV_cont(β, N, A; noise=noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
iwn = (collect(0:(N - 1)) .+ 0.5) * 2π / β * im;

# function of chi2kink
output_number = length(output_range);

# 计算积分时候网格点的权重
d = output_range[2] - output_range[1];
output_weight = fill(d, output_number);

# set the kernel matrix
kernel = Matrix{ComplexF64}(undef, N, output_number);
for i in 1:N
    for j in 1:output_number
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
function Q(u::Vector{Float64}, α::Float64)
    return α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight -
           0.5 * χ²(u)
end

# 𝞉Q/∂u
function ∂Qdiv∂u(u::Vector{Float64}, α::Float64)
    ∂Sdiv∂A = -d * (V * u)'    #行向量	
    ∂χ²div∂A = 2 / (σ^2) * (-d * G' * K + d^2 * A_vec(u)' * K' * K)    #行向量
    ∂Adiv∂u = diagm(A_vec(u)) * V
    ∂Sdiv∂u = ∂Sdiv∂A * ∂Adiv∂u
    ∂χ²div∂u = ∂χ²div∂A * ∂Adiv∂u

    return (α * ∂Sdiv∂u - ∂χ²div∂u / 2)'
end

function AFBT(f, grad, guess::Vector{Float64}; tol=1e-3, max_iter=30000)
    t = 1.0
    res = copy(guess)
    ite = 0
    while true
        if ite >= max_iter || norm(grad(res)) < tol
            println("iterations is $ite")
            return res
        end

        direct=-grad(res)/norm(grad(res))
        new_res = res+t*direct
        s = (f(res) - f(new_res) >= t^2)
        res = (1 - s) * res + s * new_res
        t = 2.0^(2 * s - 1) * t
        ite+=1
    end
end

u_opt=AFBT(u->-Q(u, α), u->-∂Qdiv∂u(u, α), zeros(n))
norm(∂Qdiv∂u(u_opt, α))

using LinearAlgebra
my_f1(x) = sum((x .- 1.1) .^ 2)
grad(x) = 2 * (x .- 1.1)
res=AFBT(my_f1, grad, zeros(40))
norm(grad(res))
