# A(w) = 1/ W |w|/sqrt{w²-Δ²}, W = 6, Δ = 0.5, Δ < |w| < W/2

using ACFlowSensitivity, CairoMakie, LinearAlgebra, Random
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

fig = plot_errorbound_cont(GFV, ctx, alg; perm=5e-5, title="Chi2kink, Cont-type, perm=5e-5")

# Get the axis from the figure and add the original A(w) line
ax = fig.content[1]
lines!(ax, ctx.mesh.mesh, A.(ctx.mesh.mesh);
       label="Origin A(ω)",
       linewidth=1,
       color=:red)
ylims!(ax, (0.0, 0.8))

# Update the legend to include the new line
axislegend(ax; position=:lt)
display(fig)

save("eb-bsc-chi2kink.pdf", fig)
