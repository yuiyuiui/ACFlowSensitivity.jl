@testset "continous_spectral_density" begin
    for T in [Float32, Float64]
        u = [T(1)]
        σ = [T(2)]
        amplitudes = [T(3)]
        A = continous_spectral_density(u, σ, amplitudes)
        @test typeof(A(T(0))) === T
        @test A(T(1)) == T(3)
    end
end

@testset "generate_GFV_cont" begin
    for T in [Float32, Float64]
        u = [T(1), T(2)]
        σ = [T(2), T(3)]
        amplitudes = [T(3), T(4)]
        N = 10
        β = T(10)
        A = continous_spectral_density(u, σ, amplitudes)
        G = generate_GFV_cont(β, N, A; noise=T(1e-4))
        @test typeof(G) === Vector{Complex{T}}
        @test length(G) == N
    end
end

@testset "generate_GFV_delta" begin
    for T in [Float32, Float64]
        M = 4
        N = 10
        poles = T.(collect(1:M))
        γ_vec = ones(T, M) ./ M
        β = T(10)
        G = generate_GFV_delta(β, N, poles, γ_vec; noise=T(1e-4))
        @test typeof(G) === Vector{Complex{T}}
        @test length(G) == N
    end
end
