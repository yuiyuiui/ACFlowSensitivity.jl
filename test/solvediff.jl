@testset "_∂χ²vecDiv∂G" begin
    T = Float64
    N = 20
    A, ctx, GFV = dfcfg(T, Cont())
    G0 = vcat(real(GFV), imag(GFV))
    pc = ACFlowSensitivity.PreComput(GFV, ctx, MaxEntChi2kink())
    ∂χ²_expect = ACFlowSensitivity._∂χ²vecDiv∂G(pc)[1]
    @test jacobian_check_v2v(G -> ACFlowSensitivity.G2χ²vec(ACFlowSensitivity.PreComput(G[1:N] +
                                                                                        im *
                                                                                        G[(N + 1):end],
                                                                                        ctx,
                                                                                        MaxEntChi2kink()))[2],
                             ∂χ²_expect, G0)
end
@testset "_∂αoptDiv∂χ²vec" begin
    T = Float64
    N = 20
    A, ctx, GFV = dfcfg(T, Cont())
    G0 = vcat(real(GFV), imag(GFV))
    pc = ACFlowSensitivity.PreComput(GFV, ctx, MaxEntChi2kink())
    _, _, χ²vec, idx = ACFlowSensitivity._∂χ²vecDiv∂G(pc)
    ∂αoptDiv∂χ²vec, _ = ACFlowSensitivity._∂αoptDiv∂χ²vec(χ²vec, pc, idx)
    gradient_check(logχ² -> ACFlowSensitivity.χ²vec2αopt(10 .^ logχ², pc.αvec[idx]),
                   vec(∂αoptDiv∂χ²vec) .* χ²vec * log(10), log10.(χ²vec))
end

#= Note: "flat" type model is not suitable for Gaussian type spectrum
"flat" may be good when iteration of MaxEntChi2kink >1 and may be more suitable for delta type spectrum
These methods and properties will be developed and tested in the future
And of course mesh type "TangentMesh" is more suitable for Gaussian type spectrum
=#

@testset "differentiation of chi2kink with specific method" begin
    jac_rtol = 3e-4
    for T in [Float32, Float64]
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        mesh, reA, ∂reADiv∂G = solvediff(GFV, ctx,
                                         MaxEntChi2kink(; model_type="Gaussian"))
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test ∂reADiv∂G isa Matrix{Complex{T}}
        G2A = G -> solve(G, ctx, MaxEntChi2kink())[2]
        @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; atol=tolerance(T), rtol=jac_rtol)
    end
end

# Note: the choices of these ηs are really mystory!!!
@testset "differentiation of chi2kink with general method" begin
    jac_rtol = 1e-1
    for T in [Float32, Float64]
        solve_tol = T == Float32 ? 2e-1 : 5.1e-3
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                alg = MaxEntChi2kink(; model_type=model_type)
                A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
                mesh, reA, ∂reADiv∂G = solvediff(GFV, ctx, alg)
                @test mesh isa Vector{T}
                @test reA isa Vector{T}
                @test ∂reADiv∂G isa Matrix{Complex{T}}
                if T === Float64
                    G2A = G -> solve(G, ctx, alg)[2]
                    @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, atol=tolerance(T),
                                             rtol=jac_rtol)
                end
            end
        end
    end
end

@testset "differentiation of BarRat with Cont spectrum" begin
    for T in [Float32, Float64]
        solve_tol = T == Float32 ? 1e-1 : 1.1e-2
        for mesh_type in [UniformMesh(), TangentMesh()]
            alg = BarRat()
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            mesh, reA, ∂reADiv∂G = solvediff(GFV, ctx, alg)
            @test mesh isa Vector{T}
            @test reA isa Vector{T}
            @test ∂reADiv∂G isa Matrix{Complex{T}}
            G2A = G -> solve(G, ctx, alg)[2]
            T == Float64 &&
                @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, atol=tolerance(T),
                                         rtol=relax_tol(T))
        end
    end
end

@testset "differentiation of BarRat with Delta spectrum" begin
    for T in [Float32, Float64] # mesh is not uesd so no test for mesh
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, BarRat())
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        if T == Float64
            @test norm(orp - p) < strict_tol(T)
            @test norm(orγ - γ) < strict_tol(T)
            G2p = G -> solve(G, ctx, BarRat())[3][1]
            G2γ = G -> solve(G, ctx, BarRat())[3][2]
            @test jacobian_check_v2v(G2p, ∂pDiv∂G, GFV)
            @test jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV)
        end
    end
end

@testset "differentiation of SSK with Delta spectrum" begin
    T = Float64
    pn = 2
    alg = SSK(pn)
    (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine)
    Random.seed!(6)
    mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    @test mesh isa Vector{T}
    @test reA isa Vector{T}
    @test p isa Vector{T}
    @test γ isa Vector{T}
    @test ∂pDiv∂G isa Matrix{Complex{T}}
    @test ∂γDiv∂G isa Matrix{Complex{T}}
    G2p = G -> solve(G, ctx, alg)[3][1]
    G2γ = G -> solve(G, ctx, alg)[3][2]
    # extremly unstable because Stoch method is moving on a grid
    # So too tiny change of input won't change the result like BarRat and MaxEnt.
    @test jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV)
    @test jacobian_check_v2v(G2p, ∂pDiv∂G, GFV; η=1e-2, rtol=0.1, show_dy=true)
end

@testset "differentiation of SAC with Delta spectrum" begin
    for T in [Float32, Float64]
        Random.seed!(6)
        pn = 2
        alg = SAC(pn)
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=alg.nfine, fp_ww=0.2,
                                     fp_mp=2.0)
        mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        @test size(∂pDiv∂G) == (pn, length(GFV))
        @test size(∂γDiv∂G) == (pn, length(GFV))
    end
end

@testset "differentiation of SOM" begin # It can run no matter its spectrumtype is Delta or Cont
    for T in [Float32, Float64]
        Random.seed!(6)
        alg = SOM()
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); fp_mp=0.3, fp_ww=0.5)
        mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        @test size(∂pDiv∂G) == (length(p), length(GFV))
        @test size(∂γDiv∂G) == (length(p), length(GFV))
    end
end

@testset "differentiation of SPX with delta spectrum, method = mean" begin
    pn = 2
    for T in [Float32, Float64]
        for method in ["mean", "best"]
            (fp_mp, fp_ww) = method == "mean" ? (2.0, 0.1) : (0.1, 0.01)
            alg = SPX(pn; method=method)
            (orp, orγ), ctx, GFV = dfcfg(T, Delta(); fp_mp=fp_mp, fp_ww=fp_ww, npole=pn)
            Random.seed!(6)
            mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
            @test mesh isa Vector{T}
            @test reA isa Vector{T}
            @test p isa Vector{T}
            @test γ isa Vector{T}
            @test ∂pDiv∂G isa Matrix{Complex{T}}
            @test ∂γDiv∂G isa Matrix{Complex{T}}
            @test size(∂pDiv∂G) == (length(p), length(GFV))
            @test size(∂γDiv∂G) == (length(p), length(GFV))
        end
    end
end

@testset "differentiation of NAC with delta spectrum" begin
    pn = 2
    for T in [Float32, Float64]
        alg = NAC(; pick=false, hardy=false)
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=2000)
        Random.seed!(6)
        mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        G2p = G -> solve(G, ctx, alg)[3][1]
        G2γ = G -> solve(G, ctx, alg)[3][2]
        @test jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV; η=1e-2, rtol=0.02)
        @test jacobian_check_v2v(G2p, ∂pDiv∂G, GFV; atol=1e-7)
    end
end

@testset "differentiation of NAC with Cont spectrum" begin
    for T in [Float32, Float64]
        rtol = T == Float32 ? 0.6 : 0.2
        alg = NAC()
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        mesh, reA, ∂reADiv∂G = solvediff(GFV, ctx, alg)
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test ∂reADiv∂G isa Matrix{Complex{T}}
        G2A = G -> solve(G, ctx, alg)[2]
        @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, rtol=rtol)
    end
end
