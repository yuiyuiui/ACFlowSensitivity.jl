using ACFlowSensitivity
using Plots, LinearAlgebra, Random

Random.seed!(6)
μ = [0.5, -2.5];
σ = [0.2, 1.0];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 801;
noise = 1e-4;
Gvalue = generate_GFV_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
iwn = (collect(0:(N-1)) .+ 0.5) * 2π / β * im;



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

u=diagm(1 ./ S) * U' * G/d

A = V * u

norm(G-K*A*d)


plot(output_range, A)
