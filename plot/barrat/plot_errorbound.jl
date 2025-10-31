include("../plot_method.jl")
alg = BarRat(; pcut=1e-1)
perm = 1e-4
plot_errorbound_delta(alg; perm=perm, mesh_type=UniformMesh(),
                      title="BarRat, Delta-type, perm = 1e-4")
# inkscape ... --export-type=pdf --export-filename=...
# inkscape /Users/syyui/projects/ACFlowSensitivity.jl/plot/maxent/eb-cont-2-chi2kink-0.0001.svg --export-type=pdf --export-filename=eb-cont-2-chi2kink-0.0001.svg