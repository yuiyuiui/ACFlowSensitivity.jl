using Zygote, LinearAlgebra, ACFlowSensitivity
include("../../test/testsetup.jl")

T = Float64
A, ctx, GFV = dfcfg_cont(T)
f(x) = solve(x, ctx, MaxEntChi2kink())[2]
jacobian(f, GFV)
