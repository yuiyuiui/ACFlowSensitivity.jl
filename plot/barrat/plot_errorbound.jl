include("../plot_method.jl")
alg = BarRat(; pcut=1e-1)
perm = 1e-4
fig = plot_errorbound_delta(alg; perm=perm, mesh_type=UniformMesh(),
                            title="BarRat, Delta-type, perm = 1e-4")
save("eb-delta-barrat.pdf", fig)
save("eb-delta-barrat.svg", fig)
