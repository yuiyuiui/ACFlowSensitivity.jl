@testset "BarRat delta pole Jacobian" begin
    ctx = CtxData(Delta(), 10.0, 20; mesh_type=UniformMesh(), σ=1e-4,
                  fp_ww=0.05, fp_mp=0.05)
    poles = [-1.0, 1.0]
    weights = [0.5, 0.5]
    G = generate_GFV_delta(10.0, 20, poles, weights)
    Random.seed!(42)
    G .+= 1e-4 .* (randn(20) .+ im .* randn(20))

    alg = BarRat(; pcut=0.1)
    _, (p0, _), (Jp, _) = solvediff(G, ctx, alg)
    order0 = sortperm(p0)

    Random.seed!(20260716)
    direction = randn(ComplexF64, 20)
    direction ./= norm(vcat(real(direction), imag(direction)))
    step = 1e-7
    _, (p1, _) = solve(G .+ step .* direction, ctx, alg)
    order1 = sortperm(p1)

    ad = real(Jp[order0, :]) * real(direction) +
         imag(Jp[order0, :]) * imag(direction)
    fd = (p1[order1] .- p0[order0]) ./ step
    @test norm(fd - ad) / norm(fd) < 1e-2
end
