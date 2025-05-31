# we directly use ∂Q/∂u and ∂²Q/∂u²
using ACFlowSensitivity, LinearAlgebra, Plots
μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 8.0;
output_number = 801;
noise = 1e-2;
Gvalue = generate_GFV_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
iwn = (collect(0:(N-1)) .+ 0.5) * 2π / β * im;



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
U, S, V = svd(K);
n = count(x -> (x >= 1e-10), S);
V = V[:, 1:n];
U = U[:, 1:n];
S = S[1:n]


# defualt model
model = exp.(-output_range .^ 2 / 4);
# 调整参数，归一化
model = model / (model' * output_weight);

# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
σ = 1e-4;

α = 1e-2;


# function Q
A_vec(u::Vector{Float64}) = model .* exp.(V * u)
χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
Q(u::Vector{Float64}, α::Float64) =
    α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight -
    0.5 * χ²(u)

# -𝞉Q/∂A
J(u::Vector{Float64}, α::Float64) =
    α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

# -∂²Q/∂A∂u
H(u::Vector{Float64}, α::Float64) =
    α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V

function newton(
    fun::Function,
    grad::Function,
    guess;
    maxiter::Int64 = 20000,
    mixing::Float64 = 0.5,
)
    function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = 1.0
        limit = 1e-4
        try
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

    f = fun(guess)
    J = grad(guess)
    back = _apply(guess, f, J)
    push!(feeds, guess)
    push!(backs, back)
    res_feed = nothing

    while true
        counter = counter + 1
        feed = feeds[end] + mixing * (backs[end] - feeds[end])

        f = fun(feed)
        J = grad(feed)
        back = _apply(feed, f, J)
        push!(feeds, feed)
        push!(backs, back)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum(abs.(back - feed)) < 1.e-4
            res_feed = feed
            break
        end
    end
    @show counter
    counter > maxiter && error("Tolerance is reached in newton()!")

    return back, counter
end

u_opt, iter_num = newton(u -> J(u, α), u -> H(u, α), zeros(n))
norm(J(u_opt, α))
println(u_opt)
for i = 1:length(u_opt)
    @show u_opt[i]
end
u0=zeros(n)
f0=J(u0, α)
println(f0)
[
    2.6352530891740856e8,
    -4.3973874413145256e8,
    -1.2456840048804651e8,
    -2.8803437377670014e8,
    -807229.5310334432,
    1.9034542985304046e7,
    -925987.6463867844,
    885949.8006538961,
    181774.63376545583,
    -34490.16347930523,
    158.72306728411806,
    1455.7028475845004,
    728.0210576916159,
    -415.3930605092992,
    -172.6347272416039,
    -24.152717693089677,
    29.100144337169503,
    -0.1489193032796473,
    4.256659921640624,
    -0.7092528540620855,
    0.07474883274188897,
    -0.042892058533315285,
    -0.017411700654695748,
    -0.0027367334177430245,
    0.0004005376066899601,
    -0.00011976793597065622,
    -3.621589920642642e-5,
    -4.206919677357116e-5,
]
J(u0, α)
norm(J(u0, α))
H(u, α)
1













# 检查牛顿法本身的正确性
using LinearAlgebra
u = rand(40)
my(x) = sum((x .- 1.1) .^ 4)
grad(x) = 4 * (x .- 1.1) .^ 3
hessel(x) = diagm(12 * (x .- 1.1) .^ 2)
sol = newton(grad, hessel, zeros(40))
norm(grad(sol[1]))



# check the corectness of ∂²Q/∂A∂u
