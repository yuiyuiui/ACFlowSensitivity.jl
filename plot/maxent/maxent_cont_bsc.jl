# A(w) = 1/ W |w|/sqrt{w²-Δ²}, W = 6, Δ = 0.5, Δ < |w| < W/2

using ACFlowSensitivity, Plots, LinearAlgebra, Random
include("../plot_method.jl")

W = 6.0
Δ = 0.5
β = 10.0
N = 20
function A(w)
    abs(w)>Δ && abs(w)<W/2 && return 1/W * abs(w) / sqrt(w^2 - Δ^2)
    return 0.0
end

Random.seed!(6)
GFV = generate_GFV_cont(β, N, A; noise=5e-5)

ctx = ACFlowSensitivity.CtxData(Cont(), β, N; mesh_bound=4, mesh_length=2000)

alg = MaxEnt(; method="chi2kink", model_type="Gaussian")

p = plot_errorbound_cont(GFV, ctx, alg; perm=1e-4, title="Chi2kink, Cont-type, perm=1e-4")

plot(p, ctx.mesh.mesh, A.(ctx.mesh.mesh); label="Origin A(w)", ylim=(0.0, 0.8))
