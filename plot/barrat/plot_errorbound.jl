include("../plot_method.jl")
alg = BarRat(; pcut=1e-1)
perm = 1e-4
plot_errorbound_delta(alg; perm=perm, mesh_type=UniformMesh(),
                      title="BarRat, Delta-type, perm = $(perm)")
