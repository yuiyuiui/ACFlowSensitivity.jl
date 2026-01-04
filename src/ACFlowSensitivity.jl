module ACFlowSensitivity
# import packages
using LinearAlgebra, Zygote, ChainRulesCore, Random, Einsum, SparseArrays
using Distributed

# export interfaces
export Mesh, make_mesh, UniformMesh, TangentMesh, Cont, Delta, Mixed
export solve, solvediff, CtxData
export SpectrumType, Cont, Delta, Mixed, SJ, BR
export Solver, BarRat, NAC, MaxEnt, SAN, SAC, SOM, SPX
export curve_fit, LsqFitResult, PronyApproximation
export fdgradient, ∇L2loss, find_peaks, pγdiff, pG2γ
export bfgs, newton

include("globalset.jl")
include("math/math.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")
include("maxentdiff.jl")
include("san.jl")
include("sac.jl")
include("som.jl")
include("spx.jl")
include("nac.jl")

include("adrules/adrules.jl")

end
