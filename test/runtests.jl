using Test,Random
using ACFlowSensitivity

Random.seed!(6)
tolerance(T) = eps(real(T))^(1//2)
strict_tol(T) = eps(real(T))^(2//3)
relax_tol(T) = eps(real(T))^(1//3)

@testset "math" begin
    include("math.jl")
end
@testset "generatedata" begin
    include("generatedata.jl")
end
@testset "mesh" begin
    include("mesh.jl")
end