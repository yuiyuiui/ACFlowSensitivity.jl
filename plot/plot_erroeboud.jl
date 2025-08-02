include("plot_method.jl")

# BarRat
alg = BarRat()
plot_errorbound_cont(alg; noise=2e-5, perm=1e-5)

# MaxEnt
alg = MaxEntChi2kink()
plot_errorbound_cont(alg; noise=1e-5)
