include("method.jl")
sst_max_mat = Matrix{Float64}(undef, 10, 3)
sst_ave_mat = Matrix{Float64}(undef, 10, 3)

FILE = "/Users/syyui/projects/ACFlowSensitivity.jl/sensitivityscale/sst.jld2"

#1 Cont
alg_vec = [BarRat(), NAC(), MaxEnt(; method="chi2kink"), MaxEnt(; method="bryan"),
           MaxEnt(; method="classic"), MaxEnt(; method="historic")]
_, ctx, GFV = dfcfg(Float64, Cont(); mesh_type=TangentMesh())
for i in 1:6
    @show "begin $i"
    sst_max, sst_ave = generate_sst(GFV, ctx, alg_vec[i])
    sst_max_mat[i, 1] = sst_max
    sst_ave_mat[i, 1] = sst_ave
    @show "end $i"
end

#=
1. generate 3 origin data

2. for every data, give a vector of alg

3. for all data, calculate sst_max and sst_ave
=#
