@testset "pγdiff" begin
    for T in [Float32, Float64] # mesh is not uesd so no test for mesh
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = ACFlowSensitivity.pγdiff(GFV, ctx, BarRat())
        @test Aout isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
    end
end

# MaxEnt
@testset "differentiation of MaxEnt Chi2kink with Cont spectrum and SJ entropy, specific method" begin
    jac_rtol = 1e-2
    alg = MaxEnt(; model_type="Gaussian", method="chi2kink")
    for T in [Float32, Float64]
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        G2A = G -> solve(G, ctx, alg)
        # @test norm(Aout - G2A(GFV)) < strict_tol(T)
        @test jcv2v0(G2A, ∂ADiv∂G, GFV, Aout; atol=tolerance(T), rtol=jac_rtol)
    end
end

@testset "differentiation of MaxEnt Chi2kink with Cont spectrum and SJ entropy, general method" begin
    jac_rtol = 1e-1
    for T in [Float32, Float64]
        solve_tol = T == Float32 ? 2e-1 : 5.1e-3
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                alg = MaxEnt(; model_type=model_type)
                A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
                Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
                @test Aout isa Vector{T}
                @test ∂ADiv∂G isa Matrix{Complex{T}}
                if T === Float64
                    G2A = G -> solve(G, ctx, alg)
                    # @test norm(Aout - G2A(GFV)) < strict_tol(T)
                    @test jcv2v0(G2A, ∂ADiv∂G, GFV, Aout; η=1e-2, atol=tolerance(T),
                                 rtol=jac_rtol)
                end
            end
        end
    end
end

@testset "differentiation of MaxEnt Chi2kink with Delta spectrum and SJ entropy" begin
    T = Float64
    alg = MaxEnt(; model_type="Gaussian", method="chi2kink")
    (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
    Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test p isa Vector{T}
    @test γ isa Vector{T}
    @test ∂pDiv∂G isa Matrix{Complex{T}}
    @test ∂γDiv∂G isa Matrix{Complex{T}}
    @test norm(orp - p) < 5e-3
    @test norm(orγ - γ) < 5e-3
    G2p = G -> solve(G, ctx, alg)[2][1]
    G2γ = G -> solve(G, ctx, alg)[2][2]
    @test jcv2v0(G2p, ∂pDiv∂G, GFV, p; atol=1e-7)
    @test jcv2v0(G2γ, ∂γDiv∂G, GFV, γ)
end

# Diff of other maxent methods for Delta spectrum is also just calling `pγdiff` just like `chi2kink` so we ignore their Delta spectrum diff tests here.
@testset "test invΛ" begin
    for T in [Float32, Float64]
        n = 10
        α = T(2.1)
        H = randn(T, n, n)
        H = H * H'
        λ = α * I(n) + H
        Λ⁻¹ = ACFlowSensitivity.invΛ(α, H)
        @test Λ⁻¹ isa Matrix{T}
        @test norm(Λ⁻¹ - inv(λ)) < strict_tol(T)
    end
end

@testset "test classicdiff with SJ entropy and Cont spectrum" begin
    for T in [Float32, Float64]
        alg = MaxEnt(; model_type="Gaussian", method="classic")
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        if T == Float64
            G2A = G -> solve(G, ctx, alg)
            @test jcv2v0(G2A, ∂ADiv∂G, GFV, Aout)
        end
    end
end

@testset "test bryandiff with SJ entropy and Cont spectrum" begin
    for T in [Float32, Float64]
        alg = MaxEnt(; model_type="Gaussian", method="bryan")
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        if T == Float64
            G2A = G -> solve(G, ctx, alg)
            @test jcv2v0(G2A, ∂ADiv∂G, GFV, Aout)
        end
    end
end

@testset "test historicdiff with SJ entropy and Cont spectrum" begin
    for T in [Float32, Float64]
        alg = MaxEnt(; model_type="Gaussian", method="historic")
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        if T == Float64
            G2A = G -> solve(G, ctx, alg)
            @test jcv2v0(G2A, ∂ADiv∂G, GFV, Aout)
        end
    end
end

# BarRat
@testset "differentiation of BarRat with Cont spectrum" begin
    for T in [Float32, Float64]
        for mesh_type in [UniformMesh(), TangentMesh()]
            alg = BarRat()
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
            @test Aout isa Vector{T}
            @test ∂ADiv∂G isa Matrix{Complex{T}}
            G2A = G -> solve(G, ctx, alg)
            T == Float64 &&
                @test jcv2v0(G2A, ∂ADiv∂G, GFV, Aout; η=1e-2, atol=tolerance(T),
                             rtol=relax_tol(T))
        end
    end
end

@testset "differentiation of BarRat with Delta spectrum" begin
    for T in [Float32, Float64] # mesh is not uesd so no test for mesh
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, BarRat())
        @test Aout isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        if T == Float64
            @test norm(orp - p) < strict_tol(T)
            @test norm(orγ - γ) < strict_tol(T)
            G2p = G -> solve(G, ctx, BarRat())[2][1]
            G2γ = G -> solve(G, ctx, BarRat())[2][2]
            @test jcv2v0(G2p, ∂pDiv∂G, GFV, p)
            @test jcv2v0(G2γ, ∂γDiv∂G, GFV, γ)
        end
    end
end

# SSK
@testset "differentiation of SSK with Delta spectrum" begin
    T = Float64
    pn = 2
    alg = SSK(pn)
    (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine)
    Random.seed!(6)
    Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test p isa Vector{T}
    @test γ isa Vector{T}
    @test ∂pDiv∂G isa Matrix{Complex{T}}
    @test ∂γDiv∂G isa Matrix{Complex{T}}
    G2p = G -> solve(G, ctx, alg)[2][1]
    G2γ = G -> solve(G, ctx, alg)[2][2]
    # extremly unstable because Stoch method is moving on a grid
    # So too tiny change of input won't change the result like BarRat and MaxEnt.
    @test jcv2v0(G2γ, ∂γDiv∂G, GFV, γ; η=1e-5, show_dy=true)
    @test jcv2v0(G2p, ∂pDiv∂G, GFV, p; η=1e-5, atol=1e-8, show_dy=true)
end

@testset "differentiation of SSK with Cont spectrum" begin
    T = Float64
    pn = 500
    alg = SSK(pn)
    A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh(), ml=2000)
    Random.seed!(6)
    Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test ∂ADiv∂G isa Matrix{Complex{T}}
    @test size(∂ADiv∂G) == (length(ctx.mesh.mesh), length(GFV))
    #G2A = G -> solve(G, ctx, alg)
    #@test jcv2v0(G2A, -∂ADiv∂G, GFV, Aout; η=1e-5, show_dy=true)
    #err = 1.6786279766194467
    #rel_err = 14.298733141006547
end

# SAC
@testset "differentiation of SAC with Delta spectrum" begin
    for T in [Float32, Float64]
        Random.seed!(6)
        pn = 2
        alg = SAC(pn)
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine, fp_ww=0.2,
                                     fp_mp=2.0)
        Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        if T == Float64
            @test size(∂pDiv∂G) == (pn, length(GFV))
            @test size(∂γDiv∂G) == (pn, length(GFV))
        end
    end
end

# SOM
@testset "differentiation of SOM with Cont spectrum" begin # It can run no matter its spectrumtype is Delta or Cont
    for T in [Float32, Float64]
        T = Float64
        Random.seed!(6)
        alg = SOM()
        A, ctx, GFV = dfcfg(T, Cont())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg; diffonly=true)
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        @test size(∂ADiv∂G) == (length(ctx.mesh.mesh), length(GFV))
    end
end

# SPX
@testset "differentiation of SPX with delta spectrum, method = mean" begin # # It can run no matter its spectrumtype is Delta or Cont
    pn = 2
    for T in [Float32, Float64]
        alg = SPX(pn; method="best")
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn)
        Random.seed!(6)
        Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        @test size(∂pDiv∂G) == (length(p), length(GFV))
        @test size(∂γDiv∂G) == (length(p), length(GFV))
        if T == Float64
            @test norm(orp - p) < 0.02
            @test norm(orγ - γ) < 0.025
        end
    end
end

# NAC
@testset "differentiation of NAC with delta spectrum" begin
    pn = 2
    for T in [Float32, Float64]
        alg = NAC(; pick=false, hardy=false)
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=2000)
        Random.seed!(6)
        Aout, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        G2p = G -> solve(G, ctx, alg)[2][1]
        G2γ = G -> solve(G, ctx, alg)[2][2]
        @test jcv2v0(G2γ, ∂γDiv∂G, GFV, γ; η=1e-2, rtol=0.02)
        @test jcv2v0(G2p, ∂pDiv∂G, GFV, p; atol=1e-7)
    end
end

@testset "differentiation of NAC with Cont spectrum" begin
    T = Float64
    alg = NAC()
    A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())

    nac0 = ACFlowSensitivity.init(GFV, ctx, alg)
    ACFlowSensitivity.run!(nac0, alg)
    Aout = ACFlowSensitivity.last(nac0)[1]
    J = Zygote.jacobian(G -> solve(G, ctx, alg, nac0), GFV)[1]

    @test J isa Matrix{Complex{T}}
    @test size(J) == (length(ctx.mesh.mesh), length(GFV))
    # G2A = G -> solve(G, ctx, alg, nac0)
    # @test jcv2v0(G2A, J, GFV, Aout; η = 1e-7)
end
