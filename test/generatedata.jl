@testset "continous_spectral_density" begin
    for T in [Float32, Float64]
        u = [T(1), T(2)]
        σ = [T(1), T(2)]
        peak = [T(1), T(2)]
        A = continous_spectral_density(u, σ, peak)
        @test A(T(1)) == T(1)
        @test A(T(2)) == T(2)
    end
end
