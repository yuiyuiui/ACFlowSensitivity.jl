@testset "_∂χ²vecDiv∂G" begin
    T = Float64
    N = 20
    A, ctx, GFV = dfcfg(T)
    G0 = vcat(real(GFV), imag(GFV))
    pc = ACFlowSensitivity.PreComput(GFV, ctx, MaxEntChi2kink())
    ∂χ²_expect = ACFlowSensitivity._∂χ²vecDiv∂G(pc)[1]
    jacobian_check_v2v(G -> ACFlowSensitivity.G2χ²vec(ACFlowSensitivity.PreComput(G[1:N] +
                                                                                  im *
                                                                                  G[(N + 1):end],
                                                                                  ctx,
                                                                                  MaxEntChi2kink()))[2],
                       ∂χ²_expect, G0)
end
@testset "_∂αoptDiv∂χ²vec" begin
    T = Float64
    N = 20
    A, ctx, GFV = dfcfg(T)
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

Besides, L2loss seems to be not suitable for sensitivity analysis. The maximum singular value `S[1]` of ∂reADiv∂G is usually huge,
which makes L2loss cone sharp at the bottom point. However, `L2loss(x,x0)` decays fast as `x` deviates from `x0` and the direction
of a random noise is different from the gradient of L2loss. The Expectation `∫_Sphere x' ∂reADiv∂G x dx` is usually much smaller than `S[1]`.
Thus L2loss is not a suitable way to measure the sensitivity. But I have not found a better way.

=#

@testset "differentiation of chi2kink with specific method" begin
    jac_rtol = 3e-4
    ∂loss_rtol = 1e-1
    for T in [Float32, Float64]
        solve_tol = T == Float32 ? 1.1e-1 : 5e-3
        A, ctx, GFV = dfcfg(T; mesh_type=TangentMesh())
        mesh, reA, ∂reADiv∂G, ∂loss = solvediff(GFV, ctx,
                                                MaxEntChi2kink(; model_type="Gaussian"))
        orA = A.(mesh)
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test ∂reADiv∂G isa Matrix{Complex{T}}
        @test ∂loss isa Vector{Complex{T}}
        @test loss(reA, orA, ctx.mesh_weights) < solve_tol
        G2A = G -> solve(G, ctx, MaxEntChi2kink())[2]
        G2l = G -> loss(reA, G2A(G), ctx.mesh_weights)
        @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; atol=tolerance(T), rtol=jac_rtol)
        @test gradient_check(G2l, ∂loss, GFV; atol=tolerance(T), rtol=∂loss_rtol, η=1e-8)
    end
end

# Note: the choices of these ηs are really mystory!!!
@testset "differentiation of chi2kink with general method" begin
    jac_rtol = 1e-1
    ∂loss_rtol = 1.5e-1
    for T in [Float32, Float64]
        solve_tol = T == Float32 ? 2e-1 : 5.1e-3
        for mesh_type in [UniformMesh(), TangentMesh()]
            for model_type in ["Gaussian", "flat"]
                alg = MaxEntChi2kink(; model_type=model_type)
                A, ctx, GFV = dfcfg(T; mesh_type=mesh_type)
                mesh, reA, ∂reADiv∂G, ∂loss = solvediff(GFV, ctx, alg)
                orA = A.(mesh)
                @test mesh isa Vector{T}
                @test reA isa Vector{T}
                @test ∂reADiv∂G isa Matrix{Complex{T}}
                @test ∂loss isa Vector{Complex{T}}
                @test loss(reA, orA, ctx.mesh_weights) < solve_tol
                if T === Float64
                    G2A = G -> solve(G, ctx, alg)[2]
                    G2l = G -> loss(reA, G2A(G), ctx.mesh_weights)
                    @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, atol=tolerance(T),
                                             rtol=jac_rtol)
                    @test gradient_check(G2l, ∂loss, GFV; atol=0,
                                         rtol=∂loss_rtol,
                                         η=5e-10)
                end
            end
        end
    end
end

# Bote :∂loss has great instability (because of the huge maximum sigular value of ∂reA) here so we don't test it
@testset "differentiation of BarRat with Cont spectrum" begin
    for T in [Float32, Float64]
        solve_tol = T == Float32 ? 1e-1 : 1.1e-2
        for mesh_type in [UniformMesh(), TangentMesh()]
            alg = BarRat(Cont())
            A, ctx, GFV = dfcfg(T; mesh_type=mesh_type)
            mesh, reA, ∂reADiv∂G, ∂loss = solvediff(GFV, ctx, alg)
            orA = A.(mesh)
            @test mesh isa Vector{T}
            @test reA isa Vector{T}
            @test ∂reADiv∂G isa Matrix{Complex{T}}
            @test ∂loss isa Vector{Complex{T}}
            @test loss(reA, orA, ctx.mesh_weights) < solve_tol
            #if T === Float64
            G2A = G -> solve(G, ctx, alg)[2]
            G2l = G -> loss(reA, G2A(G), ctx.mesh_weights)
            T == Float64 &&
                @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; η=1e-2, atol=tolerance(T),
                                         rtol=relax_tol(T))
        end
    end
end

@testset "differentiation of BarRat with Delta spectrum" begin
    for T in [Float32, Float64] # mesh is not uesd so no test for mesh
        (orp, orγ), ctx, GFV = dfcfg(T; spt=Delta(), poles_num=2)
        mesh, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, BarRat(Delta()))
        @test mesh isa Vector{T}
        @test p isa Vector{T}
        @test γ isa Vector{T}
        @test ∂pDiv∂G isa Matrix{Complex{T}}
        @test ∂γDiv∂G isa Matrix{Complex{T}}
        if T == Float64
            @test norm(orp - p) < strict_tol(T)
            @test norm(orγ - γ) < strict_tol(T)
            G2p = G -> solve(G, ctx, BarRat(Delta()))[2][1]
            G2γ = G -> solve(G, ctx, BarRat(Delta()))[2][2]
            jacobian_check_v2v(G2p, ∂pDiv∂G, GFV)
            jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV)
        end
    end
end

@testset "differentiation of SSK with Delta spectrum" begin
    Random.seed!(6)
    T = Float64
    pn = 2
    alg = SSK(pn)
    (orp, orγ), ctx, GFV = dfcfg(T; poles_num=pn, spt=Delta(), ml=alg.nfine)
    mesh, (p, γ), (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    @test mesh isa Vector{T}
    @test p isa Vector{T}
    @test γ isa Vector{T}
    @test ∂pDiv∂G isa Matrix{Complex{T}}
    @test ∂γDiv∂G isa Matrix{Complex{T}}
    @show norm(orp - p)
    @show norm(orγ - γ)
    G2p = G -> solve(G, ctx, alg)[2][1]
    G2γ = G -> solve(G, ctx, alg)[2][2]
    @test jacobian_check_v2v(G2p, ∂pDiv∂G, GFV; η=8e-3, rtol=2e-1) # extremly unstable
    @test jacobian_check_v2v(G2γ, ∂γDiv∂G, GFV)
end
