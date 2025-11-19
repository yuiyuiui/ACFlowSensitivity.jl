using Test, Random, TestExtras
using LinearAlgebra, Zygote, Distributed
using ACFlowSensitivity

Random.seed!(6)
include("testsetup.jl")

addprocs(8)
@everywhere using ACFlowSensitivity

@testset "ssk for cont" begin
    Random.seed!(6)
    T = Float64
    pn = 500
    alg = SSK(pn)
    A, ctx, GFV = dfcfg(T, Cont())
    Aout = solve(GFV, ctx, alg)
    @test Aout isa Vector{T}
    @test sum(Aout .* ctx.mesh.weight) ≈ 1.0
    @test loss(Aout, A.(ctx.mesh.mesh), ctx.mesh.weight) < 0.03
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
    G2A = G -> solve(G, ctx, alg)
    @test jacobian_check_v2v(G2A, -∂ADiv∂G, GFV; η=1e-5, show_dy=true)
end
