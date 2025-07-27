function make_model(model_type::String, ctx::CtxData{T}) where {T<:Real}
    if model_type == "Gaussian"
        f = x -> exp(-x^2/4)
    elseif model_type == "flat"
        f = x -> T(1)
    else
        error("Model $model_type not supported")
    end
    model = f.(ctx.mesh)
    model ./= sum(model .* ctx.mesh_weight)
    return model
end
