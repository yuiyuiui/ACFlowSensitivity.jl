@testset "uniform mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh = make_mesh(mb, ml, UniformMesh())
        @test typeof(mesh.mesh) === Vector{T}
        @test typeof(mesh.weight) === Vector{T}
        @test length(mesh.mesh) == ml
        @test length(mesh.weight) == ml
        @test mesh.mesh == collect(range(-mb, mb, ml))
        @test isapprox(sum(mesh.weight), mb*2, atol=tolerance(T))
    end
end

@testset "tangent mesh" begin
    ml = 100
    for T in [Float32, Float64]
        mb = T(5)
        mesh = make_mesh(mb, ml, TangentMesh())
        @test typeof(mesh.mesh) === Vector{T}
        @test typeof(mesh.weight) === Vector{T}
        @test length(mesh.mesh) == ml
        @test length(mesh.weight) == ml
        @test mesh.mesh ==
              tan.(collect(range(-T(π)/T(2.1), T(π)/T(2.1), ml)))/tan(T(π)/T(2.1))*mb
        @test isapprox(sum(mesh.weight), mb*2, atol=tolerance(T))
    end
end

@testset "SingularSpace" begin
    n = 10
    for T in [Float32, Float64]
        v1 = Vector{T}(1:n)
        v2 = Vector{Complex{T}}(1:n)
        _, ctx, GFV = dfcfg(T, Cont())
        mesh = ctx.mesh.mesh
        ss = ACFlowSensitivity.SingularSpace(GFV, ctx.iwn, mesh)
        G, K, n, U, S, V = ss
        @test typeof(ss) <: ACFlowSensitivity.SingularSpace{T}
        kernel = Matrix{Complex{T}}(undef, length(GFV), length(mesh))
        for i in 1:length(GFV)
            for j in 1:length(mesh)
                kernel[i, j] = 1 / (ctx.iwn[i] - mesh[j])
            end
        end
        G0 = vcat(real(GFV), imag(GFV))
        K0 = [real(kernel); imag(kernel)]
        @test G == G0
        @test isapprox(K, K0, atol=n*strict_tol(T))
    end
end

@testset "nearest" begin
    N = 1000000
    for T in [Float32, Float64]
        v = collect(range(T(0), T(1), N))
        r = rand(T)
        idx = findmin(abs.(v .- r))[2]
        @test idx == ACFlowSensitivity.nearest(v, r)
    end
end
