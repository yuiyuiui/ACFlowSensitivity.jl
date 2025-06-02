@testset "aaa" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        f(z) = (1) / (2 * z^2 + z + 1)
        N = 10
        train_z = [randn(T) for _ in 1:N]
        train_data = [f(z) for z in train_z]
        test_z = [randn(T) for _ in 1:N]
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

@testset "cont barrat" begin
    for T in [Float32, Float64]
        tol = T==Float64 ? 1.1e-2 : 1e-1
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                A, ctx, GFV = dfcfg_cont(T; mesh_type=mesh_type)
                mesh, reA = solve(GFV, ctx, BarRat())
                orA = A.(mesh)
                @test eltype(reA) == eltype(mesh) == T
                @test length(reA) == length(mesh) == length(ctx.mesh)
                @test loss(reA, orA, ctx.mesh_weights) < tol
            end
        end
    end
end

@testset "cont MaxEntChi2kink" begin
    for T in [Float32, Float64]
        tol = T==Float64 ? 5e-3 : 1.1e-1
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                A, ctx, GFV = dfcfg_cont(T; mesh_type=mesh_type)
                mesh, reA = solve(GFV, ctx, MaxEntChi2kink())
                orA = A.(mesh)
                @test eltype(reA) == eltype(mesh) == T
                @test length(reA) == length(mesh) == length(ctx.mesh)
                @test loss(reA, orA, ctx.mesh_weights) < tol
                @test_throws ErrorException solve(GFV, ctx, MaxEntChi2kink(; maxiter=2))
            end
        end
    end
end

#=
@testset "MaxEntChi2kink with iterative model" begin
    T = Float64
    for mesh_type in [UniformMesh(), TangentMesh()]
        for model_type in ["Gaussian", "flat"]
            A, ctx1, GFV1 = dfcfg_cont(T; noise=T(1e-3))
            mesh, reA1 = solve(GFV1, ctx1, MaxEntChi2kink())
            _, ctx2, GFV2 = dfcfg_cont(T; noise=T(1e-3))
            mesh, reA2 = solve(GFV2, ctx2, MaxEntChi2kink(; maxiter=2))
            orA = A.(mesh)
            @show error1 = loss(reA1, orA, ctx1.mesh_weights)
            @show error2 = loss(reA2, orA, ctx2.mesh_weights)
        end
    end
end
=#
