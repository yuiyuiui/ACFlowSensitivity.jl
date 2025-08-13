@testset "pγdiff" begin
    for T in [Float32, Float64] # mesh is not uesd so no test for mesh
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = ACFlowSensitivity.pγdiff(GFV, ctx, BarRat())
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
    end
end

@testset "differentiation of MaxEnt Chi2kink with Cont spectrum and SJ entropy, specific method" begin
    jac_rtol = 1e-2
    alg = MaxEnt(; model_type="Gaussian", method="chi2kink")
    for T in [Float32, Float64]
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx,
                                  MaxEnt(; model_type="Gaussian"))
        @test Aout isa Vector{T}
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        G2A = G -> solve(G, ctx, MaxEnt())
        @test jacobian_check_v2v(G2A, ∂ADiv∂G, GFV; atol=tolerance(T), rtol=jac_rtol)
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
                reA, ∂reADiv∂G = solvediff(GFV, ctx, alg)
                @test reA isa Vector{T}
                @test ∂reADiv∂G isa Matrix{Complex{T}}
                if T === Float64
                    G2A = G -> solve(G, ctx, alg)
                    @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, atol=tolerance(T),
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
    reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    @test reA isa Vector{T}
    @test p isa Vector{T}
    @test γ isa Vector{T}
    @test ∂pDiv∂G isa Matrix{Complex{T}}
    @test ∂γDiv∂G isa Matrix{Complex{T}}
    @test norm(orp - p) < 5e-3
    @test norm(orγ - γ) < 5e-3
    G2p = G -> solve(G, ctx, alg)[2][1]
    G2γ = G -> solve(G, ctx, alg)[2][2]
    @test jacobian_check_v2v(G2p, ∂pDiv∂G, GFV; atol=1e-7)
    @test jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV)
end

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

@testset "test classicdiff" begin
    for T in [Float32, Float64]
        alg = MaxEnt(; model_type="Gaussian", method="classic")
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        Aout, ∂ADiv∂G = solvediff(GFV, ctx, alg)
        @test Aout isa Vector{T}
        @test ∂ADiv∂G isa Matrix{Complex{T}}
        if T == Float64
            G2A = G -> solve(G, ctx, alg)
            @test jacobian_check_v2v(G2A, ∂ADiv∂G, GFV)
        end
    end
end

@testset "differentiation of BarRat with Cont spectrum" begin
    for T in [Float32, Float64]
        for mesh_type in [UniformMesh(), TangentMesh()]
            alg = BarRat()
            A, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type)
            reA, ∂ADiv∂G = solvediff(GFV, ctx, alg)
            @test reA isa Vector{T}
            @test ∂ADiv∂G isa Matrix{Complex{T}}
            G2A = G -> solve(G, ctx, alg)
            T == Float64 &&
                @test jacobian_check_v2v(G2A, ∂ADiv∂G, GFV; η=1e-2, atol=tolerance(T),
                                         rtol=relax_tol(T))
        end
    end
end

@testset "differentiation of BarRat with Delta spectrum" begin
    for T in [Float32, Float64] # mesh is not uesd so no test for mesh
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=2)
        reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, BarRat())
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        if T == Float64
            @test norm(orp - p) < strict_tol(T)
            @test norm(orγ - γ) < strict_tol(T)
            G2p = G -> solve(G, ctx, BarRat())[2][1]
            G2γ = G -> solve(G, ctx, BarRat())[2][2]
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
    reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    @test reA isa Vector{T}
    @test p isa Vector{T}
    @test γ isa Vector{T}
    @test ∂pDiv∂G isa Matrix{Complex{T}}
    @test ∂γDiv∂G isa Matrix{Complex{T}}
    G2p = G -> solve(G, ctx, alg)[2][1]
    G2γ = G -> solve(G, ctx, alg)[2][2]
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
        reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
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
        reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        @test size(∂pDiv∂G) == (length(p), length(GFV))
        @test size(∂γDiv∂G) == (length(p), length(GFV))
    end
end

@testset "differentiation of SPX with delta spectrum, method = mean" begin # # It can run no matter its spectrumtype is Delta or Cont
    pn = 2
    for T in [Float32, Float64]
        alg = SPX(pn; method="best")
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn)
        Random.seed!(6)
        reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        @test size(∂pDiv∂G) == (length(p), length(GFV))
        @test size(∂γDiv∂G) == (length(p), length(GFV))
    end
end

@testset "differentiation of NAC with delta spectrum" begin
    pn = 2
    for T in [Float32, Float64]
        alg = NAC(; pick=false, hardy=false)
        (orp, orγ), ctx, GFV = dfcfg(T, Delta(); npole=pn, ml=2000)
        Random.seed!(6)
        reA, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
        @test reA isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        G2p = G -> solve(G, ctx, alg)[2][1]
        G2γ = G -> solve(G, ctx, alg)[2][2]
        @test jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV; η=1e-2, rtol=0.02)
        @test jacobian_check_v2v(G2p, ∂pDiv∂G, GFV; atol=1e-7)
    end
end

@testset "differentiation of NAC with Cont spectrum" begin
    for T in [Float32, Float64]
        rtol = T == Float32 ? 0.6 : 0.2
        alg = NAC()
        A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh())
        reA, ∂reADiv∂G = solvediff(GFV, ctx, alg)
        @test reA isa Vector{T}
        @test ∂reADiv∂G isa Matrix{Complex{T}}
        G2A = G -> solve(G, ctx, alg)
        @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, rtol=rtol)
    end
end
