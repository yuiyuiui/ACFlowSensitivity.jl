@testset "aaa" begin
    N = 10
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        f(z) = (1) / (2 * z^2 + z + 1)
        train_z = [randn(T) for _ in 1:N]
        train_data = [f(z) for z in train_z]
        test_z = [randn(T) for _ in 1:N]
        test_data = [f(z) for z in test_z]
        brf, _ = ACFlowSensitivity.aaa(train_z, train_data; alg=BarRat())
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
        p = ACFlowSensitivity.bc_poles(w, g)
        @test p isa Vector{Complex{T}}
        @test isapprox(p, [-2, 2], atol=strict_tol(T))
    end
end

# w = [-3/2,4,-3/2], g = [1,0,-1], v = [-1,0,1]
# f = 1.5/(x-2) + 1.5/(x+2)
@testset "Poles" begin
    N = 20
    for T in [Float32, Float64]
        CT = Complex{T}
        w = CT.([-1.5, 4, -1.5])
        g = CT.([1, 0, -1])
        v = CT.([-1, 0, 1])
        r = ACFlowSensitivity.BarRatFunc(w, g, v)
        iwn = collect(1:N) * T(1)im
        GFV = r.(iwn)
        poles = ACFlowSensitivity.Poles(GFV, iwn, 1e-3)
        p, γ = poles(w, g)
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test isapprox(p, [-2, 2], atol=strict_tol(T))
        @test isapprox(γ, [1.5, 1.5], atol=strict_tol(T))
    end
end

@testset "delta barrat" begin
    for T in [Float32, Float64]
        (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        mesh, reA, (rep, reγ) = solve(GFV, ctx, BarRat())
        @test reA isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        T == Float64 && @test norm(poles - rep) < strict_tol(T)
        T == Float64 && @test norm(γ - reγ) < strict_tol(T)
    end
end

@testset "cont barrat" begin
    for T in [Float32, Float64]
        tol = T==Float32 ? 1e-1 : 1.1e-2
        for mesh_type in [UniformMesh(), TangentMesh()]
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            mesh, reA = solve(GFV, ctx, BarRat())
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
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            for prony_tol in [0, (T==Float32 ? 1e-4 : 1e-8)]
                mesh, reA = solve(GFV, ctx,
                                  BarRat(; denoisy=true, prony_tol=prony_tol))
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
                A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
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
			A, ctx1, GFV1 = dfcfg(T, Cont(); noise=T(1e-3))
			mesh, reA1 = solve(GFV1, ctx1, MaxEntChi2kink())
			_, ctx2, GFV2 = dfcfg(T, Cont(); noise=T(1e-3))
			mesh, reA2 = solve(GFV2, ctx2, MaxEntChi2kink(; maxiter=2))
			orA = A.(mesh)
			@show error1 = loss(reA1, orA, ctx1.mesh_weights)
			@show error2 = loss(reA2, orA, ctx2.mesh_weights)
		end
	end
end
=#

@testset "serve functions in ssk" begin
    pn = 2
    T = Float64
    Random.seed!(1234)
    (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=pn)
    alg = SSK(pn)
    fine_mesh = collect(range(ctx.mesh[1], ctx.mesh[end], alg.nfine)) # ssk needs high-precise linear grid
    MC = @constinferred ACFlowSensitivity.init_mc(alg)
    SE = @constinferred ACFlowSensitivity.init_element(alg, MC.rng, ctx)
    SC = @constinferred ACFlowSensitivity.init_context(SE, GFV, fine_mesh, ctx, alg)
    Aout, _, _ = @constinferred ACFlowSensitivity.run!(MC, SE, SC, alg)
end

# I don't test type stability of ssk for Float32 because it often fails to reach equilibrium state.
# But in some random case I don't record it does succeed and the result is type stable.
@testset "ssk for delta" begin # It can run no matter its spectrumtype is Delta or Cont. Cont is slow for sac and we don't have accuracy need now. So ignore Cont.
    Random.seed!(6)
    T = Float64
    pn = 2
    alg = SSK(pn)
    # It's recommended to use large mesh length for ssk. But limited by the poles searching ability of `pind_peaks`, I temporarily set it only the default value 801
    (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine)
    mesh, Aout, (rep, reγ) = solve(GFV, ctx, alg)
    @test mesh isa Vector{T}
    @test Aout isa Vector{T}
    @test rep isa Vector{T}
    @test reγ isa Vector{T}
    @test norm(poles - rep) < 5 * relax_tol(T)
    @test norm(γ - reγ) == 0
end

@testset "sac for delta" begin # It can run no matter its spectrumtype is Delta or Cont. Cont is slow for sac and we don't have accuracy need now. So ignore Cont.
    for T in [Float32, Float64]
        Random.seed!(6)
        pn = 2
        alg = SAC(pn)
        (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine, fp_ww=0.2,
                                     fp_mp=2.0)
        mesh, Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        @test norm(poles - rep) < 0.25
        @test norm(γ - reγ) == 0
    end
end

@testset "som for delta" begin
    for T in [Float32, Float64]
        Random.seed!(6)
        alg = SOM()
        A, ctx, GFV = dfcfg(T, Delta(); fp_mp=0.3, fp_ww=0.5)
        mesh, Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
    end
end

@testset "som for cont" begin
    for T in [Float32, Float64]
        Random.seed!(6)
        alg = SOM()
        # It's recommended to use large mesh length for ssk. But limited by the poles searching ability of `pind_peaks`, I temporarily set it only the default value 801
        A, ctx, GFV = dfcfg(T, Cont())
        mesh, Aout = solve(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test Aout isa Vector{T}
        @test loss(Aout, A.(mesh), ctx.mesh_weights) < 0.5
    end
end
