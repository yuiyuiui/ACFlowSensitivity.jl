module ACFlowSensitivity
# import packages
using LinearAlgebra, Zygote, QuadGK, Optim, LsqFit
using Zygote:@adjoint



# export interfaces
export my_newton,my_GD_v1,my_GD_v2, my_curve_fit
export continous_spectral_density, kernel,aaa_check,kernel
export discrete_GF,benchmark_GF
export generate_G_values_cont,generate_G_values_disc,AD_aaa
export DireInverse_check,generate_G_values_delta,DireInverse
export ADaaa,Solver,get_loss,aaa_cont_FiniDIff_Chain
export ADaaaBase,GiwnToL0,GiwnL0ToLoss,Loss
export my_chi2kink, ADchi2kink



# `include` other source files into this module
include("Method.jl")


end