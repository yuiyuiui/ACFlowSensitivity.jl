include("../plot_method.jl")
alg = BarRat(; pcut=1e-1)
perm = 5e-5
fig = plot_errorbound_delta(alg; perm=perm, mesh_type=UniformMesh(),
                            title="BarRat, Delta-type, perm = 5e-5")
save("eb-delta-barrat.pdf", fig)
