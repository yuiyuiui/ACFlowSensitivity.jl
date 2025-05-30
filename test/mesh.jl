@testset "uniform mesh" begin
    opn = 100
    for T in [Float32, Float64]
        opb = T(5)
        mesh, mesh_weights = make_mesh(opb,opn,UniformMesh{T}())
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weights) === Vector{T}
        @test length(mesh) == opn
        @test length(mesh_weights) == opn
        @test mesh == collect(range(-opb,opb,opn))
        @test isapprox(sum(mesh_weights), opb*2, atol=tolerance(T))
    end
end

@testset "tangent mesh" begin
    opn = 100
    for T in [Float32, Float64]
        opb = T(5)
        mesh, mesh_weights = make_mesh(opb,opn,TangentMesh(T(2.1)))
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weights) === Vector{T}
        @test length(mesh) == opn
        @test length(mesh_weights) == opn
        @test mesh == tan.(collect(range(-T(π)/T(2.1), T(π)/T(2.1), opn)))/tan(T(π)/T(2.1))*opb
        @test isapprox(sum(mesh_weights), opb*2, atol=tolerance(T))
    end
end