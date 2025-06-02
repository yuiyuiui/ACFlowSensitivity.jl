module ACFlowSensitivity
# import packages
using LinearAlgebra #, Zygote
#using Zygote: @adjoint

# export interfaces
export continous_spectral_density, generate_GFV_cont, generate_GFV_delta
export make_mesh, UniformMesh, TangentMesh
export solve, CtxData
export BarRat, MaxEntChi2kink
#=
export my_newton,my_GD_v1,my_GD_v2, my_curve_fit
export continous_spectral_density, kernel,aaa_check,kernel
export discrete_GF,benchmark_GF
export AD_aaa
export DireInverse_check,generate_GFV_delta,DireInverse
export ADaaa,Solver,get_loss,aaa_cont_FiniDIff_Chain
export ADaaaBase,GiwnToL0,GiwnL0ToLoss,Loss
export my_chi2kink, ADchi2kink
=#

# `include` other source files into this module
include("math.jl")
include("generaldata.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")
#=
include("ADaaa.jl")
include("maxent.jl")
include("sac.jl")
include("ADsac.jl")
=#

end
