include("../plot_method.jl")

# MaxEnt
alg = MaxEnt(; method="chi2kink", model_type="Gaussian")
plot_errorbound_cont(alg; noise=1e-5, nwave=1)
plot_errorbound_cont(alg; noise=1e-5, nwave=2)
plot_errorbound_cont(alg; noise=1e-5, perm=1e-5, nwave=3)

plot_errorbound_delta(alg; perm=1e-6)
