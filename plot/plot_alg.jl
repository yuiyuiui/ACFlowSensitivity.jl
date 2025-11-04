include("plot_method.jl")

# BarRat
alg = BarRat()
plot_alg_cont(alg; nwave=1)
plot_alg_cont(alg; nwave=2)
plot_alg_cont(alg; nwave=3, noise_num=2)

alg = BarRat()
plot_alg_delta(alg)

# MaxEnt
alg = MaxEnt(; method="chi2kink", model_type="Gaussian")
plot_alg_cont(alg; nwave=1, noise_num=5)
plot_alg_cont(alg; nwave=2, noise_num=5)
plot_alg_cont(alg; nwave=3, noise_num=5)

alg = MaxEnt(; method="chi2kink", model_type="flat", stype=BR())
plot_alg_delta(alg)

alg = MaxEnt(; method="bryan", model_type="flat", stype=BR())
plot_alg_delta(alg)

alg = MaxEnt(; method="classic", model_type="flat", stype=BR())
plot_alg_delta(alg)

alg = MaxEnt(; method="historic", model_type="flat")
plot_alg_delta(alg)
#=
(poles, γ), ctx, GFV = dfcfg(Float64, Delta(); npole=2)
alg = MaxEnt(; method="historic", model_type="flat")
reA, (rep,reγ) = solve(GFV, ctx, alg)
plot(ctx.mesh.mesh, reA)
=#

# SSK
alg = SSK(500)
plot_alg_cont(alg; noise_num=1)

alg = SSK(2)
plot_alg_delta(alg)

# SAC
alg = SAC(512)
plot_alg_cont(alg)

alg = SAC(2)
plot_alg_delta(alg; fp_ww=0.2, fp_mp=2.0)

# SPX
alg = SPX(2; method="mean", ntry=100)
plot_alg_cont(alg)

alg = SPX(2; method="best")
plot_alg_delta(alg)

# SOM
alg = SOM()
plot_alg_cont(alg)

alg = SOM()
plot_alg_delta(alg; fp_ww=0.07, fp_mp=1.05)

# NAC
alg = NAC()
plot_alg_cont(alg)

alg = NAC(; pick=false, hardy=false, eta=1e-4)
plot_alg_delta(alg)
