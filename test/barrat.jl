@testset "aaa" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        f(z) = (1) / (2 * z^2 + z + 1)
        N = 10
        train_z = [randn(T) for _ = 1:N]
        train_data = [f(z) for z in train_z]
        test_z = [randn(T) for _ = 1:N]
        test_data = [f(z) for z in test_z]
        w, g, v = ACFlowSensitivity.aaa(train_z, train_data; alg=BarRat())
        @test w isa Vector{T}
        @test g isa Vector{T}
        @test v isa Vector{T}
        res = zero(test_data)
        for i in eachindex(test_z)
            res[i] += sum((w .* v) ./ (test_z[i] .- g)) / sum(w ./ (test_z[i] .- g))
        end
        @test res isa Vector{T}
        @test isapprox(res, test_data, atol=strict_tol(T))
    end
end

@testset "cont BarRat" begin
    for T in [Float32, Float64]
        A, ctx, GFV = dfcfg_cont(T)
        mesh, reA = solve(GFV, ctx, BarRat())
        orA = A.(mesh)
        @test eltype(reA) == eltype(mesh) == T
        @test length(reA) == length(mesh) == length(ctx.mesh)
        T == Float64 && @test loss(reA, orA, ctx.mesh_weights) < 1e-2
    end
end
