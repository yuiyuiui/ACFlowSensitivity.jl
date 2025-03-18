using Plots, Statistics
using LinearAlgebra, ACFlowSensitivity, Random

Random.seed!(8)
noise_vec = [1e-5, 5 * 1e-5, 1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2]
T = 3
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
iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im
d = output_range[2] - output_range[1]
η = 1e-8
error_mean = zeros(length(noise_vec))
error_std = zeros(length(noise_vec))

i = 4
noise = noise_vec[i]

error = zeros(T)

for t in 1:T
	@show t
	Gvalue = generate_G_values_cont(β, N, A; noise = noise)
	Aout = my_chi2kink(iwn, Gvalue, output_range)
	ADAout = ACFlowSensitivity.ADchi2kink_v2(iwn, Gvalue, output_range)
	δ = sum(exp.(my_chi2kink(iwn, Gvalue + η * ADAout, output_range) - Aout)) * d - output_number * d
	δ1 = η * norm(ADAout)^2
	error[t] = abs(δ - δ1) / max(δ, δ1)
	@show error[t], δ, δ1
end

error_mean[i] = mean(error)
error_std[i] = std(error)


1
0.0030932242380227147

0.0012950785395133327
##

2

0.0004653487670203175

6.717130649196088e-5

##

3
0.0004196720641048725

4.305635524177369e-5
##

4
0.0009361987458679409

0.0002250864649115637

##

5
0.0011170149400416157

0.0009276081795913588


##

6
0.014348344568973492

0.016032447577238592
##

7
6.937490578592872e-7

1.9855626501135279e-7


error_mean = [0.0030932242380227147, 0.0004653487670203175, 0.0004196720641048725, 0.0009361987458679409, 0.0011170149400416157, 0.014348344568973492, 6.937490578592872e-7]
error_std = [0.0012950785395133327, 6.717130649196088e-5, 4.305635524177369e-5, 0.0002250864649115637, 0.0009276081795913588, 0.016032447577238592, 1.9855626501135279e-7]





# 绘制误差条图
plot(noise_vec, error_mean, yerr = error_std,
	xlabel = "Noise Scale", ylabel = "Error rate",
	title = "Error rate of AD and FD",
	legend = false,
	xticks = (noise_vec, string.(noise_vec)),
	linewidth = 2,
	markersize = 5,
	markerstrokewidth = 0.5,
	markerstrokecolor = :black)

	savefig("error_bar_plot.png")