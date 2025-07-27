module ACFlowSensitivity
# import packages
using LinearAlgebra, Zygote, Random

# export interfaces
export make_mesh, UniformMesh, TangentMesh, Cont, Delta, Mixed
export solve, solvediff, CtxData
export SpectrumType, Cont, Delta, Mixed
export BarRat, NAC, MaxEntChi2kink, SSK, SAC, SOM, SPX
export curve_fit, LsqFitResult, PronyApproximation
export fdgradient, ∇L2loss
export bfgs, newton

include("globalset.jl")
include("math/math.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")
include("ssk.jl")
include("sac.jl")
include("som.jl")
include("spx.jl")
include("nac.jl")

include("adrules/adrules.jl")

end
