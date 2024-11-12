using ACFlow, Plots, DelimitedFiles, Test

@testset "correct aaa" begin
	# continous spectral density
	function continous_spectral_density(μ::Vector{Float64}, σ::Vector{Float64}, peak::Vector{Float64})
		@assert length(μ) == length(σ) == length(peak)
		n = length(μ)
		function y(x::Float64)
			res = 0
			for i ∈ 1:n
				res += peak[i] * exp(-(x - μ[i])^2 / (2 * σ[i]^2))
			end
			return res
		end
		return y
	end

	# Generate observed green function values with noise and continuous spectral density
	function generate_observed_GFV_cont(A, β, N; noise = 1e-3, step = 1e-3, boundary = 20)
		grid = (collect(0:N-1) .+ 0.5) * 2π / β
		Int_stp_num = trunc(Int64, boundary / step)
		n = length(grid)
		res = zeros(ComplexF64, n)

		for i ∈ 1:n
			# 计算每个 \hat{G}(w_n)=\int_R A(x)/(iw_n-x)dx

			# 辛普森法的积分
			integral = 0.0
			for j ∈ -Int_stp_num:Int_stp_num
				x = j * step
				coeff = 1.0

				# 使用辛普森法的权重
				if j == -Int_stp_num || j == Int_stp_num
					coeff = 1  # 端点的权重为 1
				elseif j % 2 == 0
					coeff = 2  # 偶数点的权重为 2
				else
					coeff = 4  # 奇数点的权重为 4
				end

				integral += coeff * A(x) / (im * grid[i] - x)
			end

			integral *= step / 3  # 最后乘以步长 / 3
			res[i] = integral
		end

		for i in 1:length(res)
			res[i] += res[i] * noise * rand() * exp(2π * rand() * im)
		end
		return res
	end

	function reconstruct_spectral_density(A, β, N, output_bound, output_number, noise)

		iwn = collect((0:N-1) .+ 0.5) * 2π / β
		GFV = generate_observed_GFV_cont(A, β, N; noise = noise)

		B = Dict{String, Any}(
			"solver" => "BarRat",  # Choose MaxEnt solver
			"mtype" => "gauss",   # Default model function
			"mesh" => "tangent", # Mesh for spectral function
			"ngrid" => N,        # Number of grid points for input data
			"nmesh" => output_number,       # Number of mesh points for output data
			"wmax" => output_bound,       # Right boundary of mesh
			"wmin" => -output_bound,      # Left boundary of mesh
			"beta" => β,      # Inverse temperature
		)

		S = Dict{String, Any}(
			"atype" => "cont",
			#"denoise"=>"prony_o",
			"denoise" => "none",
			#"denoise"=>"prony_s",
			"epsilon" => 1e-10,
			"pcut" => 1e-3,
			"eta" => 1e-2,
		)
		setup_param(B, S)

		mesh, reA, _ = solve(iwn, GFV)
		return mesh, reA

	end

	μ = [0.5, -2.5]
	σ = [0.2, 0.8]
	peak = [1.0, 0.3]
	A = continous_spectral_density(μ, σ, peak)
	β = 10.0
	N = 20
	output_bound = 5.0
	output_number = 801

	avrage_diff = 0.0
	times = 10
    Amesh=Float64[]
	for i = 1:times
		noise = 1e-6
		Amesh, reA = reconstruct_spectral_density(A, β, N, output_bound, output_number, noise)
		avrage_diff += abs.((reA-A.(Amesh))[1:end-1])' *
					   (Amesh[2:end] - Amesh[1:end-1])
	end
	avrage_diff /= times
	av_diff_ratio = avrage_diff / ((A.(Amesh)[1:end-1])' * (Amesh[2:end] - Amesh[1:end-1]))

	@test av_diff_ratio < 0.1
end







