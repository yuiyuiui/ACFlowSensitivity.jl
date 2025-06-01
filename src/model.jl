function make_model(model_type::String, ctx::CtxData{T}) where {T<:Real}
    if model_type == "Gaussian"
        f = x->exp(-x^4)
        fmesh = f.(ctx.mesh)
        c = sum(fmesh .* ctx.mesh_weights)
        return fmesh/c
    elseif model_type == "flat"
        return ones(T, length(ctx.mesh)) ./ sum(ctx.mesh_weights)
    else
        error("Model $model_type not supported")
    end
end
