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
    size(J)[1] == 1 && (J = vec(J))
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

"""
    second_derivative(x::AbstractVector, y::AbstractVector)

Compute second derivative y''(x). If the length of `x` and `y` is `N`,
the length of the returned vector is `N-2`.

### Arguments
* x -> Real frequency mesh.
* y -> Function values at real axis.

### Returns
* val -> y''(x).
"""
function second_derivative(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)

    N = length(x)
    h₁ = view(x, 2:(N - 1)) - view(x, 1:(N - 2))
    h₂ = view(x, 3:N) - view(x, 2:(N - 1))

    y_forward = view(y, 3:N)
    y_mid = view(y, 2:(N - 1))
    y_backward = view(y, 1:(N - 2))

    num = h₁ .* y_forward + h₂ .* y_backward - (h₁ + h₂) .* y_mid
    den = (h₂ .^ 2) .* h₁ + (h₁ .^ 2) .* h₂
    return 2 .* num ./ den
end
