include("method.jl")

alg_num = 8
T = Float64
S = Complex{T}

sst_cont = Vector{T}(undef, alg_num)
sst_delta = Vector{T}(undef, alg_num)
sst_mixed = Vector{T}(undef, alg_num)
chi2_cont = Vector{T}(undef, alg_num)
chi2_delta = Vector{T}(undef, alg_num)
chi2_mixed = Vector{T}(undef, alg_num)

Aout_cont = Vector{Vector{T}}(undef, alg_num)
Aout_delta = Vector{Vector{T}}(undef, alg_num)
Aout_mixed = Vector{Vector{T}}(undef, alg_num)
p_delta = Vector{T}(undef, alg_num)
γ_delta = Vector{T}(undef, alg_num)

J_cont = Vector{Matrix{S}}(undef, alg_num)
Jp_delta = Vector{Matrix{S}}(undef, alg_num)
Jγ_delta = Vector{Matrix{S}}(undef, alg_num)
J_mixed = Vector{Matrix{S}}(undef, alg_num)

method_name_vec = ["BarRat", "NAC", "Chi2kink", "Bryan", "Classic", "Historic", "SSK",
                   "SAC"]

#1 Cont
alg_cont_vec = [BarRat(), NAC(), MaxEnt(; method="chi2kink"), MaxEnt(; method="bryan"),
                MaxEnt(; method="classic"), MaxEnt(; method="historic"), SSK(500), SAC(512)]
Acont, ctx_cont, GFV_cont = dfcfg(Float64, Cont(); mesh_type=TangentMesh(), noise=1e-5,
                                  ml=2000)
Aor_cont = Acont.(ctx_cont.mesh.mesh)
ctx_cont_vec = [deepcopy(ctx_cont) for i in 1:alg_num]

for i in 1:alg_num
    @show "begin $i"
    Random.seed!(6)
    sst_cont[i], chi2_cont[i], Aout_cont[i], J_cont[i] = generate_sst(GFV_cont,
                                                                      ctx_cont_vec[i],
                                                                      alg_cont_vec[i])
    @show "end $i"
end

#2 Delta
alg_delta_vec = [BarRat(), NAC(; pick=false, hardy=false, eta=1e-4),
                 MaxEnt(; method="chi2kink", model_type="flat", stype=BR()),
                 MaxEnt(; method="bryan", model_type="flat", stype=BR()),
                 MaxEnt(; method="classic", model_type="flat", stype=BR()),
                 MaxEnt(; method="historic", model_type="flat"), SSK(2), SAC(2)]
(por, γor), ctx_delta, GFV_delta = dfcfg(Float64, Delta(); npole=2, mb=4, ml=2000)
ctx_delta_vec = CtxData[]
# BarRat, NAC
push!(ctx_delta_vec, deepcopy(ctx_delta))
push!(ctx_delta_vec, deepcopy(ctx_delta))
# Chi2kink, Bryan, Classic
ctx_delta_maxent = dfcfg(Float64, Delta(); npole=2, mb=4, ml=2000, fp_ww=0.1, fp_mp=1.0)[2]
push!(ctx_delta_vec, deepcopy(ctx_delta_maxent))
push!(ctx_delta_vec, deepcopy(ctx_delta_maxent))
push!(ctx_delta_vec, deepcopy(ctx_delta_maxent))
push!(ctx_delta_vec, deepcopy(ctx_delta_maxent))

# SSK
push!(ctx_delta_vec, deepcopy(ctx_delta))

# SAC
ctx_delta_sac = dfcfg(Float64, Delta(); npole=2, fp_ww=0.2, fp_mp=5.0, mb=4, ml=2000)[2]
push!(ctx_delta_vec, deepcopy(ctx_delta_sac))

for i in 1:alg_num
    @show "begin $i"
    Random.seed!(6)
    sst_delta[i], chi2_delta[i], Aout_delta[i], (p_delta[i], γ_delta[i]), (Jp_delta[i], Jγ_delta[i]) = generate_sst(GFV_delta,
                                                                                                                    ctx_delta_vec[i],
                                                                                                                    alg_delta_vec[i])
    @show "end $i"
end

# 3. Mixed
W = 6.0
Δ = 0.5
β = 10.0
N = 20
function Amixed(w)
    abs(w)>Δ && abs(w)<W/2 && return 1/W * abs(w) / sqrt(w^2 - Δ^2)
    return 0.0
end

GFV_mixed = generate_GFV_cont(β, N, Amixed; noise=5e-5)

alg_mixed_vec = [BarRat(), NAC(; eta=1e-4), MaxEnt(; method="chi2kink"),
                 MaxEnt(; method="bryan"),
                 MaxEnt(; method="classic"), MaxEnt(; method="historic"), SSK(500),
                 SAC(512)]
ctx_mixed = dfcfg(Float64, Cont(); mb=4, ml=2000)[2]
Aor_mixed = Amixed.(ctx_mixed.mesh.mesh)
ctx_mixed_vec = [deepcopy(ctx_mixed) for i in 1:alg_num]

for i in 1:alg_num
    @show "begin $i"
    Random.seed!(6)
    sst_mixed[i], chi2_mixed[i], Aout_mixed[i], J_mixed[i] = generate_sst(GFV_mixed,
                                                                          ctx_mixed_vec[i],
                                                                          alg_vec[i])
    @show "end $i"
end

for j in 1:alg_num
    record_cont("Cont_$(method_name_vec[j])", sst_cont[j], chi2_cont[j], Aout_cont[j],
                J_cont[j], Aor_cont, GFV_cont, ctx_cont_vec[j], alg_cont_vec[j])
end

for j in 1:alg_num
    record_delta("./sensitivityscale/results/Delta_$(method_name_vec[j])", sst_delta[j],
                 chi2_delta[j], Aout_delta[j], (p_delta[i], γ_delta[i]),
                 (Jp_delta[i], Jγ_delta[i]), (por, γor), GFV_delta, ctx_delta_vec[j],
                 alg_delta_vec[j])
end

for j in 1:alg_num
    record_mixed("./sensitivityscale/results/Mixed_$(method_name_vec[j])", sst_mixed[j],
                 chi2_mixed[j], Aout_mixed[j], J_mixed[j], Aor_mixed, GFV_mixed,
                 ctx_mixed_vec[j], alg_mixed_vec[j])
end
