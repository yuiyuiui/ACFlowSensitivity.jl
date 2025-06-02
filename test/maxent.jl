#@testset "cont BarRat" begin
#for T in [Float32, Float64]
T = Float64
A, ctx, GFV = dfcfg_cont(T)
mesh, reA = solve(GFV, ctx, MaxEntChi2kink())
orA = A.(mesh)
@test eltype(reA) == eltype(mesh) == T
@test length(reA) == length(mesh) == length(ctx.mesh)
T == Float64 && @test loss(reA, orA, ctx.mesh_weights) < 1e-2
#end
#end

using Plots
plot(mesh, A.(mesh))
plot!(mesh, reA)
