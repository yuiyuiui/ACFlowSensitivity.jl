#@testset "differentiation of chi2kink" begin
    #for T in [Float32, Float64]
    T = Float64
    rtol = T == Float32 ? 1e-1 : 1e-2
    tol = T==Float32 ? 1.1e-1 : 5e-3
        A, ctx, GFV = dfcfg_cont(T;noise = 1e-4)
        mesh, reA, ∂reADiv∂G, ∂loss = solvediff(GFV, ctx, MaxEntChi2kink())
        orA = A.(mesh)
        @test mesh isa Vector{T}
        @test reA isa Vector{T}
        @test ∂reADiv∂G isa Matrix{Complex{T}}
        @test ∂loss isa Vector{Complex{T}}
        @test loss(reA, orA, ctx.mesh_weights) < tol
        G2A = G -> solve(G, ctx, MaxEntChi2kink())[2]
        G2l = G -> loss(reA,G2A(G), ctx.mesh_weights)
        fdgradient(G2l, GFV)
        @test jacobian_check_v2v(G2A, ∂reADiv∂G, GFV; atol = tolerance(T), rtol = rtol)
        @test gradient_check(G2l, ∂loss, GFV; atol = tolerance(T), rtol = rtol)

    end
end