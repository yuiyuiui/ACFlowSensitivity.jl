include("../plot_method.jl")

# MaxEnt
alg = MaxEnt(; method="chi2kink", model_type="Gaussian")

plot_errorbound_cont(alg; noise=1e-5, nwave=2, title="Chi2kink, Cont-type, perm=1e-4",
                     mesh_type=TangentMesh())
save("eb-cont-chi2kink.pdf", fig)