using ACFlowSensitivity
using Plots, LinearAlgebra, Random, Test

@testset "ADMaxEnt" begin
	# loss function = ||A1 -A0||_2

	Random.seed!(4)
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
	d = output_range[2] - output_range[1]
	iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im

	noise = 1e-2
	Gvalue = generate_G_values_cont(β, N, A; noise = noise)
	Aout = my_chi2kink(iwn, Gvalue, output_range)

	η = 1e-8
	_, dlossdivdG = ADchi2kink(iwn, Gvalue, output_range)
	δ = norm(my_chi2kink(iwn,Gvalue + η * dlossdivdG, output_range) - Aout)*sqrt(d)
    δ1= η *norm(dlossdivdG)^2
	norm(δ-δ1)/max(norm(δ),norm(δ1))

	isapprox(δ, δ1, rtol = 0.01)
end

