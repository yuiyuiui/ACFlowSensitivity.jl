using Plots, Statistics
using LinearAlgebra, ACFlowSensitivity, Random

Random.seed!(4)
noise_vec = [1e-5, 5 * 1e-5, 1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2]
T = 5
μ = [0.5, -2.5]
σ = [0.2, 1.0]
peak = [1.0, 0.3]
A = continous_spectral_density(μ, σ, peak)
β = 10.0
N = 20
output_bound = 5.0
output_number = 801
output_range = range(-output_bound, output_bound, output_number)
output_range = collect(output_range)
iwn = (collect(0:(N-1)) .+ 0.5) * 2π / β * im
d = output_range[2] - output_range[1]
η = 1e-8
error_mean = zeros(length(noise_vec))
error_std = zeros(length(noise_vec))

i = 6
noise = noise_vec[i]

error = zeros(T)

for t = 1:T
    @show t
    Gvalue = generate_GFV_cont(β, N, A; noise = noise)
    Aout = my_chi2kink(iwn, Gvalue, output_range)
    _, dlossdivdG = ADchi2kink(iwn, Gvalue, output_range)
    δ = norm(my_chi2kink(iwn, Gvalue + η * dlossdivdG, output_range) - Aout) * sqrt(d)
    δ1 = η * norm(dlossdivdG)^2
    error[t] = abs(δ - δ1) / max(δ, δ1)
    @show error[t], δ, δ1
end


error_mean[i] = mean(error)
error_std[i] = std(error)


# we delete some invalid date when i = 2,3


# 绘制误差条图
plot(
    noise_vec,
    error_mean,
    yerr = error_std,
    xlabel = "Noise Scale",
    ylabel = "Error rate",
    title = "Error rate of AD and FD, loss function = ||A1 -A0||_2",
    legend = false,
    xscale = :log10,
    xticks = (noise_vec, string.(noise_vec)),
    linewidth = 2,
    markersize = 5,
    markerstrokewidth = 0.5,
    markerstrokecolor = :black,
)

savefig("error_bar_plot.png")
