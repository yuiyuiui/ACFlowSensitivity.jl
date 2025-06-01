@testset "model" begin
    for T in [Float32, Float64]
        β = T(10)
        N = 10
        for mesh_type in [TangentMesh(), UniformMesh()]
            ctx = CtxData(β, N)
            for model_type in ["Gaussian", "flat"]
                model = ACFlowSensitivity.make_model(model_type, ctx)
                @test model isa Vector{T}
                @test isapprox(sum(model .* ctx.mesh_weights), T(1), atol=strict_tol(T))
            end
        end
    end
end
