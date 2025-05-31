using Test, Random
using ACFlowSensitivity

Random.seed!(6)
include("testsetup.jl")

@testset "math" begin
    include("math.jl")
end
@testset "generatedata" begin
    include("generatedata.jl")
end
@testset "mesh" begin
    include("mesh.jl")
end
@testset "barrat" begin
    include("barrat.jl")
end
