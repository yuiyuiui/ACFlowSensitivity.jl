@testset "uniform mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh, mesh_weights = make_mesh(mb, ml, UniformMesh{T}())
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weights) === Vector{T}
        @test length(mesh) == ml
        @test length(mesh_weights) == ml
        @test mesh == collect(range(-mb, mb, ml))
        @test isapprox(sum(mesh_weights), mb*2, atol = tolerance(T))
    end
end

@testset "tangent mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh, mesh_weights = make_mesh(mb, ml, TangentMesh(T(2.1)))
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weights) === Vector{T}
        @test length(mesh) == ml
        @test length(mesh_weights) == ml
        @test mesh ==
              tan.(collect(range(-T(π)/T(2.1), T(π)/T(2.1), ml)))/tan(T(π)/T(2.1))*mb
        @test isapprox(sum(mesh_weights), mb*2, atol = tolerance(T))
    end
end
