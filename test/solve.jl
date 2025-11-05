# BarRat
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
        Aout, (rep, reγ) = solve(GFV, ctx, BarRat())
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        T == Float64 && @test norm(poles - rep) < strict_tol(T)
        T == Float64 && @test norm(γ - reγ) < strict_tol(T)
    end
end

@testset "cont barrat" begin
    for T in [Float32, Float64]
        tol = T == Float32 ? 2e-1 : 1e-2
        for mesh_type in [UniformMesh(), TangentMesh()]
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            Aout = solve(GFV, ctx, BarRat())
            @test eltype(Aout) == T
            @test length(Aout) == length(ctx.mesh.mesh)
            T == Float64 && @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < tol
        end
    end
end

@testset "prony barrat" begin
    for T in [Float32, Float64]
        tol = T == Float32 ? 3e-1 : 1e-1
        for mesh_type in [UniformMesh(), TangentMesh()]
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            for prony_tol in [0, (T == Float32 ? 1e-4 : 1e-8)]
                Aout = solve(GFV, ctx,
                             BarRat(; denoisy=true, prony_tol=prony_tol))
                @test eltype(Aout) == T
                @test length(Aout) == length(ctx.mesh.mesh)
                @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < tol
            end
        end
    end
end

# MaxEnt
@testset "cont MaxEnt SJ entropy" begin
    tol = 0.1
    T = Float32
    A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
    Aout = solve(GFV, ctx, MaxEnt(; model_type="Gaussian"))
    @test eltype(Aout) == T
    @test length(Aout) == length(ctx.mesh.mesh)
    @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < tol

    T = Float64
    for method in ["chi2kink", "classic", "bryan", "historic"]
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
                Aout = solve(GFV, ctx, MaxEnt(; model_type=model_type, method=method))
                @test eltype(Aout) == T
                @test length(Aout) == length(ctx.mesh.mesh)
                lsres = loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight)
                @test lsres < tol
            end
        end
    end
end

@testset "delta MaxEnt Chi2kink SJ entropy" begin
    for T in [Float64, Float32]
        alg = MaxEnt(; model_type="Gaussian")
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        if T == Float64
            @test norm(orp - rep) < 5e-3
            @test norm(orγ - reγ) < 5e-3
        end
    end
end

# Test BR entropy
# T = Float32 would cause too much numerical instability, so we don't test it here.
@testset "maxent BR entropy with Cont spectrum" begin
    T = Float64
    for method in ["chi2kink", "classic", "bryan", "historic"]
        alg = MaxEnt(; model_type="Gaussian", stype=BR(), method=method)
        _, ctx, GFV = dfcfg(T, Cont(); μ=[T(0)], σ=[T(1)], amplitudes=[T(1)])
        Aout = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
    end
end

# SSK
@testset "serve functions in ssk" begin
    pn = 2
    T = Float64
    Random.seed!(1234)
    (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=pn)
    alg = SSK(pn)
    fine_mesh = collect(range(ctx.mesh.mesh[1], ctx.mesh.mesh[end], alg.nfine)) # ssk needs high-precise linear grid
    MC = @constinferred ACFlowSensitivity.init_mc(alg)
    SE = @constinferred ACFlowSensitivity.init_element(alg, MC.rng, ctx)
    SC = @constinferred ACFlowSensitivity.init_context(SE, GFV, fine_mesh, ctx, alg)
    Aout, _, _ = @constinferred ACFlowSensitivity.run!(MC, SE, SC, alg)
end

# I don't test type stability of ssk for Float32 because it often fails to reach equilibrium state.
# But in some random case I don't record it does succeed and the result is type stable.
@testset "ssk for delta" begin
    Random.seed!(6)
    T = Float64
    pn = 2
    alg = SSK(pn)
    # It's recommended to use large mesh length for ssk. But limited by the poles searching ability of `pind_peaks`, I temporarily set it only the default value 801
    (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine)
    Aout, (rep, reγ) = solve(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test rep isa Vector{T}
    @test reγ isa Vector{T}
    @test norm(poles - rep) < 1e-4
    @test norm(γ - reγ) < 1e-5
end

@testset "ssk for cont" begin
    Random.seed!(6)
    T = Float64
    pn = 500
    alg = SSK(pn)
    A, ctx, GFV = dfcfg(T, Cont())
    Aout = solve(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test sum(Aout .* ctx.mesh.weight) ≈ 1.0
    @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.1
end

# SAC
@testset "sac for delta" begin
    for T in [Float32, Float64]
        Random.seed!(6)
        pn = 2
        alg = SAC(pn)
        (poles, γ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine, fp_ww=0.2,
                                     fp_mp=2.0)
        Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        @test norm(poles - rep) < 0.25
        @test norm(γ - reγ) < 0.1
    end
end

@testset "sac for cont" begin
    Random.seed!(6)
    T = Float64
    pn = 512
    alg = SAC(pn)
    A, ctx, GFV = dfcfg(T, Cont())
    Aout = solve(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.1
end

# SOM
@testset "som for delta" begin
    pn = 2
    for T in [Float32, Float64]
        Random.seed!(6)
        alg = SOM()
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); fp_mp=1.25, fp_ww=0.07, npole=pn, mb=T(5))
        Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        if T == Float64
            @show norm(orp - rep)
            @show norm(orγ - reγ)
        end
    end
end

@testset "som for cont" begin
    for T in [Float32, Float64]
        Random.seed!(6)
        alg = SOM()
        # It's recommended to use large mesh length for som. But limited by the poles searching ability of `pind_peaks`, I temporarily set it only the default value 801
        A, ctx, GFV = dfcfg(T, Cont())
        Aout = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test abs(sum(Aout .* ctx.mesh.weight) - 1) < 1e-3
        @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.5
    end
end

# SPX
# It's recommended to use best method for spx.
@testset "spx for delta with best " begin
    pn = 2
    for T in [Float32, Float64]
        alg = SPX(pn; method="best")
        (poles, γ), ctx, GFV = dfcfg(T, Delta(); fp_mp=0.1, npole=pn, ml=alg.nfine)
        Random.seed!(6)
        Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        @test norm(rep - poles) < 0.02
    end
end

@testset "spx for cont with mean " begin
    pn = 2
    T = Float64
    alg = SPX(pn; method="mean", ntry=100)
    A, ctx, GFV = dfcfg(T, Cont())
    Random.seed!(6)
    Aout = solve(GFV, ctx, alg)
    @test Aout isa Vector{T}
    #@test abs(sum(Aout .* ctx.mesh.weight) - 1.0) < 0.1
    #@test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.1
end

# NAC
@testset "nac for cont" begin
    for T in [Float32, Float64]
        alg = T == Float32 ? NAC(; hardy=false) : NAC()
        # It's recommended to use large mesh length for ssk. But limited by the poles searching ability of `pind_peaks`, I temporarily set it only the default value 801
        A, ctx, GFV = dfcfg(T, Cont())
        Aout = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        T == Float64 && @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.05
    end
end

@testset "nac for delta" begin
    for T in [Float32, Float64]
        alg = NAC(; pick=false, hardy=false, eta=1e-4)
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); mb=T(5))
        Aout, (rep, reγ) = solve(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test rep isa Vector{T}
        @test reγ isa Vector{T}
        @test norm(orp - rep) < 0.007
        @test norm(orγ - reγ) < 0.011
    end
end
