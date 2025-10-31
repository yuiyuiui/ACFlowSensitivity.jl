include("../plot_method.jl")

# MaxEnt
alg = MaxEnt(; method="chi2kink", model_type="Gaussian")
plot_errorbound_cont(alg; noise=1e-5, nwave=1, mesh_type=TangentMesh())
plot_errorbound_cont(alg; noise=1e-5, nwave=2, title="Chi2kink, Cont-type, perm=1e-4", mesh_type=TangentMesh())
plot_errorbound_cont(alg; noise=1e-5, perm=1e-5, nwave=3, mesh_type=TangentMesh())

plot_errorbound_delta(alg; perm=1e-6, mesh_type=UniformMesh())
