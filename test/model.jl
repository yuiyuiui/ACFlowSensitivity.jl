@testset "model" begin
    for T in [Float32, Float64]
        β = T(10)
        N = 10
        for mesh_type in [TangentMesh(), UniformMesh()]
            ctx = CtxData(Cont(), β, N; mesh_type=mesh_type)
            for model_type in ["Gaussian", "flat"]
                if model_type == "Gaussian"
                    f = x->exp(-x^2/4)
                elseif model_type == "flat"
                    f = x->T(1)
                else
                    error("Invalid model type")
                end
                res = f.(ctx.mesh)
                res .= res ./ sum(res .* ctx.mesh_weights)
                model = ACFlowSensitivity.make_model(model_type, ctx)
                @test res == model
                @test model isa Vector{T}
                @test isapprox(sum(model .* ctx.mesh_weights), T(1), atol=strict_tol(T))
            end
        end
    end
end
