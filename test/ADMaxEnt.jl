using ACFlowSensitivity
using Plots, LinearAlgebra, Random, Test

@testset "ADMaxEnt_v2" begin

	Random.seed!(6)
	μ = [0.5, -2.5]
	σ = [0.2, 1.0]
	peak = [1.0, 0.3]
	A = continous_spectral_density(μ, σ, peak)
	β = 10.0
	N = 20
	output_bound = 5.0
	output_number = 801
    d = output_range[2] - output_range[1]
    iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im

	noise = 1e-4
	Gvalue = generate_G_values_cont(β, N, A; noise = noise)
	Aout = my_chi2kink(iwn, Gvalue, output_range)

	η = 1e-8
	ADAout_v2 = ACFlowSensitivity.ADchi2kink_v2(iwn, Gvalue, output_range)
	δ = sum(exp.(my_chi2kink(iwn,Gvalue + η * ADAout_v2, output_range) - Aout))*d - output_number*d
    δ1= η *norm(ADAout_v2)^2

	isapprox(δ, δ1, rtol = 1e-3)
end
