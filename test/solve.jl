@testset "aaa" begin
    N = 10
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        f(z) = (1) / (2 * z^2 + z + 1)
        train_z = [randn(T) for _ in 1:N]
        train_data = [f(z) for z in train_z]
        test_z = [randn(T) for _ in 1:N]
        test_data = [f(z) for z in test_z]
        brf, _ = ACFlowSensitivity.aaa(train_z, train_data; alg=BarRat(Cont()))
        @test brf isa ACFlowSensitivity.BarRatFunc{T}
        @test brf.w isa Vector{T}
        @test brf.g isa Vector{T}
        @test brf.v isa Vector{T}
        res = brf.(test_z)
        @test res isa Vector{T}
        @test isapprox(res, test_data, atol=strict_tol(T))
    end
end

@testset "bc_poles" begin
    for T in [Float32, Float64]
        w = T.([-1.5, 4, -1.5]) .+ 0im
        g = T.([1, 0, -1]) .+ 0im
        v = rand(T, 3) .+ 0im
        r = ACFlowSensitivity.BarRatFunc(w, g, v)
        p = ACFlowSensitivity.bc_poles(r)
        @test p isa Vector{Complex{T}}
        @test isapprox(p, [-2, 2], atol=strict_tol(T))
    end
end

# w = [-3/2,4,-3/2], g = [1,0,-1], v = [-1,0,1]
# f = 1.5/(x-2) + 1.5/(x+2)
@testset "poles" begin
    N = 20
    for T in [Float32, Float64]
        CT = Complex{T}
        w = CT.([-1.5, 4, -1.5])
        g = CT.([1, 0, -1])
        v = CT.([-1, 0, 1])
        r = ACFlowSensitivity.BarRatFunc(w, g, v)
        iwn = collect(1:N) * T(1)im
        GFV = r.(iwn)
        p, γ = ACFlowSensitivity.poles(GFV, r, iwn, 1e-3)
        @test p isa Vector{Complex{T}}
        @test isapprox(p, [-2, 2], atol=strict_tol(T))
        @test isapprox(γ, [1.5, 1.5], atol=strict_tol(T))
    end
end

@testset "cont barrat" begin
    for T in [Float32, Float64]
        tol = T==Float32 ? 1e-1 : 1.1e-2
        for mesh_type in [UniformMesh(), TangentMesh()]
            A, ctx, GFV = dfcfg_cont(T; mesh_type=mesh_type)
            mesh, reA = solve(GFV, ctx, BarRat(Cont()))
            orA = A.(mesh)
            @test eltype(reA) == eltype(mesh) == T
            @test length(reA) == length(mesh) == length(ctx.mesh)
            @test loss(reA, orA, ctx.mesh_weights) < tol
        end
    end
end

@testset "prony barrat" begin
    for T in [Float32, Float64]
        tol = T==Float32 ? 3e-1 : 1e-1
        for mesh_type in [UniformMesh(), TangentMesh()]
            A, ctx, GFV = dfcfg_cont(T; mesh_type=mesh_type)
            for prony_tol in [0, (T==Float32 ? 1e-4 : 1e-8)]
                mesh, reA = solve(GFV, ctx,
                                  BarRat(Cont(); denoisy=true, prony_tol=prony_tol))
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
        tol = T==Float32 ? 2e-1 : 5.1e-3
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                A, ctx, GFV = dfcfg_cont(T; mesh_type=mesh_type)
                mesh, reA = solve(GFV, ctx, MaxEntChi2kink(model_type=model_type))
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
