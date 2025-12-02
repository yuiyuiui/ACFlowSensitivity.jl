using Test, Random, TestExtras
using LinearAlgebra, Zygote
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
@testset "solvediff" begin
    include("solvediff.jl")
end
@testset "adrule" begin
    include("adrules/adrules.jl")
end
