module ACFlowSensitivity
# import packages
using LinearAlgebra

# export interfaces
export continous_spectral_density, kernel,aaa_check
export discrete_GF,benchmark_GF
export Lp_norm


# `include` other source files into this module
include("Method.jl")
include("SelfDefMathMethod.jl")

end