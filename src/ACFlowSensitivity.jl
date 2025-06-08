module ACFlowSensitivity
# import packages
using LinearAlgebra, Zygote

# export interfaces
export continous_spectral_density, generate_GFV_cont, generate_GFV_delta
export make_mesh, UniformMesh, TangentMesh, Cont, Delta, Mixed
export solve, solvediff, CtxData
export BarRat, MaxEntChi2kink
export curve_fit, LsqFitResult
export fdgradient

include("math.jl")
include("generaldata.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")

include("adrules/adrules.jl")

end
