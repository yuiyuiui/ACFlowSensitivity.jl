include("method.jl")
using JLD2
# sst_mat = Matrix{Float64}(undef, 3, 10)

FILE = "/Users/syyui/projects/ACFlowSensitivity.jl/sensitivityscale/sst.jld2"

@load FILE sst_mat

#1 Cont
alg_vec = [BarRat(), NAC(), MaxEnt(; method="chi2kink"), MaxEnt(; method="bryan"),
           MaxEnt(; method="classic"), MaxEnt(; method="historic")]
_, ctx, GFV = dfcfg(Float64, Cont(); mesh_type=TangentMesh(), noise=1e-5)
for i in 1:6
    @show "begin $i"
    i==2 && continue
    sst_mat[1, i] = generate_sst(deepcopy(GFV), deepcopy(ctx), deepcopy(alg_vec[i]))
    @show "end $i"
end

alg_s = [SSK(500), SAC(512), SOM(), SPX(2; method="mean", ntry=100)]
for i in 1:4
    @show "begin $i"
    sst_mat[1, i+6] = generate_sst(deepcopy(GFV), deepcopy(ctx), deepcopy(alg_s[i]))
    @show "end $i"
end

#2 Delta
alg_vec = [BarRat(), NAC(; pick=false, hardy=false, eta=1e-4),
           MaxEnt(; method="chi2kink", model_type="flat", stype=BR()),
           MaxEnt(; method="bryan", model_type="flat", stype=BR()),
           MaxEnt(; method="classic", model_type="flat", stype=BR()),
           MaxEnt(; method="historic", model_type="flat"), SSK(2), SAC(2), SOM(),
           SPX(2; method="best")]
_, ctx0, GFV = dfcfg(Float64, Delta(); npole=2)
ctx_vec = CtxData[]
# BarRat, NAC, MaxEnt(chi2kink, bryan, classic, historic), SSK
for _ in 1:7
    push!(ctx_vec, deepcopy(ctx0))
end
# SAC
ctx_sac = dfcfg(Float64, Delta(); npole=2, fp_ww=0.2, fp_mp=2.0)[2]
push!(ctx_vec, deepcopy(ctx_sac))
# SOM
ctx_som = dfcfg(Float64, Delta(); fp_ww=0.07, fp_mp=1.05)[2]
push!(ctx_vec, deepcopy(ctx_som))
# SPX
push!(ctx_vec, deepcopy(ctx0))
for i in 1:10
    @show "begin $i"
    sst_mat[2, i] = generate_sst(deepcopy(GFV), deepcopy(ctx_vec[i]), deepcopy(alg_vec[i]))
    @show "end $i"
end

# 3. Mixed
W = 6.0
Δ = 0.5
β = 10.0
N = 20
function A(w)
    abs(w)>Δ && abs(w)<W/2 && return 1/W * abs(w) / sqrt(w^2 - Δ^2)
    return 0.0
end

Random.seed!(6)
GFV = generate_GFV_cont(β, N, A; noise=5e-5)

alg_vec = [BarRat(), NAC(; eta=1e-4), MaxEnt(; method="chi2kink"), MaxEnt(; method="bryan"),
           MaxEnt(; method="classic"), MaxEnt(; method="historic"), SSK(500), SAC(512),
           SOM(), SPX(2; method="mean", ntry=100)]
ctx = dfcfg(Float64, Cont(); mb=4, ml=2000)[2]
for i in 1:10
    @show "begin $i"
    sst_mat[3, i] = generate_sst(deepcopy(GFV), deepcopy(ctx), deepcopy(alg_vec[i]))
    @show "end $i"
end

@save FILE sst_mat

#=
1. generate 3 origin data

2. for every data, give a vector of alg

3. for all data, calculate sst_max and sst_ave
=#
