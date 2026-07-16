@testset "SAC sensitivity moments respect truncated SVD rank" begin
    ctx = CtxData(Cont(), 10.0, 4;
                  mesh_bound=4.0, mesh_length=101,
                  mesh_type=UniformMesh(), σ=1.0e-4)
    G = ComplexF64[1 / (im * wn - 1.0) for wn in ctx.wn]
    alg = SAC(16; nfine=100, nwarm=1, nstep=1, nalph=2, nchain=1)
    mc = ACFlowSensitivity.init_mc(alg)
    element = ACFlowSensitivity.init_element(alg, mc.rng, Float64)
    fine_mesh = collect(range(ctx.mesh.mesh[1], ctx.mesh.mesh[end], alg.nfine))
    context = ACFlowSensitivity.init_context(element, G, fine_mesh, ctx, alg)

    @test size(context.U, 1) == 2length(G)
    @test size(context.U, 2) == length(context.Gᵥ)
    @test size(context.Kor, 1) == length(context.Gᵥ)
    @test size(context.E1, 1) == length(context.Gᵥ)
    @test size(context.E2, 2) == length(context.Gᵥ)

    positions = view(element.Γₚ, :, 1)
    amplitudes = view(element.Γₐ, :, 1)
    @test length(context.Kor[:, positions] * amplitudes) == size(context.E1, 1)
end
