using Test, ACFlowSensitivity,LinearAlgebra

@testset "ADChi2kink" begin
	μ = [0.5, -2.5]
	σ = [0.2, 0.8]
	peak = [1.0, 0.3]
	A = continous_spectral_density(μ, σ, peak)
	β = 10.0
	N = 20
	output_bound = 5.0
	output_number = 401
	noise = 0.0
	Gvalue = generate_G_values_cont(β, N, A; noise = noise)
	output_range = range(-output_bound, output_bound, output_number)
	output_range = collect(output_range)
	iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im

	Aout0 = my_chi2kink(iwn, Gvalue, output_range)
	ADAout = ADchi2kink(iwn, Gvalue, output_range)
	
    
    noise1 = 1e-2
	Gvalue1 = generate_G_values_cont(β, N, A; noise = noise1)
    Aout1 = my_chi2kink(iwn, Gvalue1, output_range)

    d = output_range[2] - output_range[1]
    @test norm(Aout0-Aout1)*sqrt(d) < noise1 * norm(ADAout)


end