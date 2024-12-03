module ACFlowSensitivity
# import packages
using LinearAlgebra, Zygote, QuadGK
using Zygote:@adjoint



# export interfaces
export continous_spectral_density, kernel,aaa_check,kernel
export discrete_GF,benchmark_GF
export generate_G_values_cont,AD_aaa
export DireInverse_check,generate_G_values_delta,DireInverse
export ADaaa,Solver,get_loss,aaa_cont_FiniDIff_Chain
export ADaaaBase,GiwnToL0,GiwnL0ToLoss,Loss



# `include` other source files into this module
include("Method.jl")


end