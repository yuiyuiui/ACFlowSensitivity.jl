module ACFlowSensitivity
# import packages
using LinearAlgebra, Zygote, Random

# export interfaces
export continous_spectral_density, generate_GFV_cont, generate_GFV_delta
export make_mesh, UniformMesh, TangentMesh, Cont, Delta, Mixed
export solve, solvediff, CtxData
export SpectrumType, Cont, Delta, Mixed
export BarRat, MaxEntChi2kink, SSK, SAC, SOM, SPX
export curve_fit, LsqFitResult, PronyApproximation
export fdgradient, ∇L2loss

include("math/math.jl")
include("generaldata.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")
include("ssk.jl")
include("sac.jl")
include("som.jl")
include("spx.jl")

include("adrules/adrules.jl")

end
