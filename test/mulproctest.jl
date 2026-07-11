using Test, Random, TestExtras
using LinearAlgebra, Zygote, Distributed
using ACFlowSensitivity

Random.seed!(6)
include("testsetup.jl")

addprocs(8)
@everywhere using ACFlowSensitivity

@testset "ssk for cont" begin
    T = Float64
    pn = 500
    A, ctx, GFV = dfcfg(T, Cont())

    Random.seed!(6)
    alg0 = SSK(pn; nchain=9, nstep=25, nwarm=10)
    Aout0 = solve(GFV, ctx, alg0)
    @test Aout0 isa Vector{T}
    @test sum(Aout0 .* ctx.mesh.weight) ≈ 1.0

    Random.seed!(6)
    alg1 = SSK(pn)
    Aout1 = solve(GFV, ctx, alg1)
    @test Aout1 isa Vector{T}
    @test sum(Aout1 .* ctx.mesh.weight) ≈ 1.0
    @test loss(Aout1, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.03
end

@testset "differentiation of SSK with Cont spectrum" begin
    T = Float64
    pn = 500
    A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh(), ml=2000)

    Random.seed!(6)
    alg0 = SSK(pn; nchain=9, nstep=25, nwarm=10)
    Aout0, ∂ADiv∂G0 = solvediff(GFV, ctx, alg0)
    @test Aout0 isa Vector{T}
    @test ∂ADiv∂G0 isa Matrix{Complex{T}}
    @test size(∂ADiv∂G0) == (length(ctx.mesh.mesh), length(GFV))

    Random.seed!(6)
    alg1 = SSK(pn)
    Aout1, ∂ADiv∂G1 = solvediff(GFV, ctx, alg1)
    @test Aout1 isa Vector{T}
    @test ∂ADiv∂G1 isa Matrix{Complex{T}}
    @test size(∂ADiv∂G1) == (length(ctx.mesh.mesh), length(GFV))
    G2A = G -> solve(G, ctx, alg1)
    @test jcv2v0(G2A, ∂ADiv∂G1, GFV, Aout1; η=1e-5, rtol=2.0)
end

@testset "sac for cont" begin
    T = Float64
    pn = 512
    A, ctx, GFV = dfcfg(T, Cont())

    Random.seed!(6)
    alg0 = SAC(pn; nchain=9, nstep=4000)
    Aout0 = solve(GFV, ctx, alg0)
    @test Aout0 isa Vector{T}

    Random.seed!(6)
    alg1 = SAC(pn)
    Aout1 = solve(GFV, ctx, alg1)
    @test Aout1 isa Vector{T}
    @test loss(Aout1, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.08
end

@testset "differentiation of SAC with Cont spectrum" begin
    T = Float64
    pn = 500
    A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh(), ml=2000)

    Random.seed!(6)
    alg0 = SAC(pn; nchain=9, nstep=4000)
    Aout0, ∂ADiv∂G0 = solvediff(GFV, ctx, alg0)
    @test Aout0 isa Vector{T}
    @test ∂ADiv∂G0 isa Matrix{Complex{T}}
    @test size(∂ADiv∂G0) == (length(ctx.mesh.mesh), length(GFV))

    Random.seed!(6)
    alg1 = SAC(pn)
    Aout1, ∂ADiv∂G1 = solvediff(GFV, ctx, alg1)
    @test Aout1 isa Vector{T}
    @test ∂ADiv∂G1 isa Matrix{Complex{T}}
    @test size(∂ADiv∂G1) == (length(ctx.mesh.mesh), length(GFV))
    G2A = G -> solve(G, ctx, alg1);
    @test jcv2v0(G2A, ∂ADiv∂G1, GFV, Aout1; η=1e-6, rtol=2.0)
end
