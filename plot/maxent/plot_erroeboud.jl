include("../plot_method.jl")

# MaxEnt
alg = MaxEnt(; method="chi2kink", model_type="Gaussian")

fig = plot_errorbound_cont(alg; noise=1e-5, nwave=2, perm=5e-5,
                           title="Chi2kink, Cont-type, perm=5e-5",
                           mesh_type=TangentMesh())
save("eb-cont-chi2kink.pdf", fig)
