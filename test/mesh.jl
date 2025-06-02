@testset "uniform mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh, mesh_weights = make_mesh(mb, ml, UniformMesh())
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weights) === Vector{T}
        @test length(mesh) == ml
        @test length(mesh_weights) == ml
        @test mesh == collect(range(-mb, mb, ml))
        @test isapprox(sum(mesh_weights), mb*2, atol=tolerance(T))
    end
end

@testset "tangent mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh, mesh_weights = make_mesh(mb, ml, TangentMesh())
        @test typeof(mesh) === Vector{T}
        @test typeof(mesh_weights) === Vector{T}
        @test length(mesh) == ml
        @test length(mesh_weights) == ml
        @test mesh ==
              tan.(collect(range(-T(π)/T(2.1), T(π)/T(2.1), ml)))/tan(T(π)/T(2.1))*mb
        @test isapprox(sum(mesh_weights), mb*2, atol=tolerance(T))
    end
end

@testset "SingularSpace" begin
    n = 10
    for T in [Float32, Float64]
        v1 = Vector{T}(1:n)
        v2 = Vector{Complex{T}}(1:n)
        _, ctx, GFV = dfcfg_cont(T)
        ss = ACFlowSensitivity.SingularSpace(GFV, ctx.mesh, ctx.iwn)
        G, K, n, U, S, V = ss
        @test typeof(ss) <: ACFlowSensitivity.SingularSpace{T}
        kernel = Matrix{Complex{T}}(undef, length(GFV), length(ctx.mesh))
        for i in 1:length(GFV)
            for j in 1:length(ctx.mesh)
                kernel[i, j] = 1 / (ctx.iwn[i] - ctx.mesh[j])
            end
        end
        G0 = vcat(real(GFV), imag(GFV))
        K0 = [real(kernel); imag(kernel)]
        @test G == G0
        @test isapprox(K, K0, atol=n*strict_tol(T))
    end
end
