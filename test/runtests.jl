using Test, Random, LinearAlgebra, TestExtras
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
@testset "model" begin
    include("model.jl")
end
@testset "solve" begin
    include("solve.jl")
end
