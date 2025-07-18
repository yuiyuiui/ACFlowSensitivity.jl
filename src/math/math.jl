tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

include("integral.jl")
include("optim.jl")
#-----------------------------------------

#= for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0], [ε], [1 / (sqrt(2π) * ε)])
end
=#

# calculate jacobian with finite-difference. Borrowed form https://github.com/yuiyuiui/ACFlow
# it accepts function that maps vector to vector or number
Base.vec(x::Number) = [x]
function fdgradient(f::Function, x::Vector{T}) where {T<:Number}
    J = zeros(T, length(f(x)), length(x))
    rel_step = cbrt(eps(real(eltype(x))))
    abs_step = rel_step
    @inbounds for i in 1:length(x)
        xₛ = x[i]
        ϵ = max(rel_step * abs(xₛ), abs_step)
        x[i] = xₛ + ϵ
        y₂ = vec(f(x))
        x[i] = xₛ - ϵ
        y₁ = vec(f(x))
        J[:, i] .+= (y₂ - y₁) ./ (2 * ϵ)
        x[i] = xₛ
    end
    T<:Complex && @inbounds for i in 1:length(x)
        xₛ = x[i]
        ϵ = max(rel_step * abs(xₛ), abs_step)
        x[i] = xₛ + im * ϵ
        y₂ = vec(f(x))
        x[i] = xₛ - im * ϵ
        y₁ = vec(f(x))
        J[:, i] .+= im * (y₂ - y₁) ./ (2 * ϵ)
        x[i] = xₛ
    end
    return J
end

function ∇L2loss(J::Matrix{T}, w::Vector{R}) where {T<:Number,R<:Real}
    @assert R == real(T)
    n = size(J, 2)
    Dsw = Diagonal(sqrt.(w))
    _, S, V = svd(Dsw * hcat(real(J), imag(J)))
    T<:Real && return S[1], V[1:n, 1] * S[1]
    return S[1], (V[1:n, 1] + im * V[(n + 1):2n, 1]) * S[1]
end

include("prony.jl")
include("statistic.jl")
include("poles.jl")
include("interpolation.jl")
