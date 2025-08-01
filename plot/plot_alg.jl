include("plot_method.jl")

# BarRat
alg = BarRat()
plot_alg_cont(alg)

alg = BarRat()
plot_alg_delta(alg)

# MaxEntChi2kink
alg = MaxEntChi2kink()
plot_alg_cont(alg; noise_num=5)

alg = MaxEntChi2kink(; model_type="flat")
plot_alg_delta(alg)

# SSK
alg = SSK(500)
plot_alg_cont(alg)

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
plot_alg_delta(alg; fp_ww=0.02, fp_mp=0.9)

# NAC
alg = NAC()
plot_alg_cont(alg)

alg = NAC(; pick=false, hardy=false)
plot_alg_delta(alg)
