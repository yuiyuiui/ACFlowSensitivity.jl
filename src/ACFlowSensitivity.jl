module ACFlowSensitivity
# import packages
using LinearAlgebra

# export interfaces
export continous_spectral_density, generate_GFV_cont, generate_GFV_delta
export make_mesh, UniformMesh, TangentMesh
export solve, CtxData
export BarRat, MaxEntChi2kink
export curve_fit, LsqFitResult

include("math.jl")
include("generaldata.jl")
include("mesh.jl")
include("solve.jl")
include("barrat.jl")
include("model.jl")
include("maxent.jl")

end
