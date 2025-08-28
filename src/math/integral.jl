function integral(f::Function, a::T, b::T; h::T=T(1e-4)) where {T<:Real}
    n_raw = floor((b - a) / h)
    n = Int(n_raw)
    if isodd(n)
        n -= 1
    end
    if n < 2
        error("step is too large")
    end

    fa = f(a)
    !(typeof(fa) <: Union{T,Complex{T}}) &&
        error("Type of the output of f should be consistent with its input")
    fb = f(a + h * T(n))
    acc = fa + fb

    @inbounds for i in 1:(n - 1)
        x = a + h * T(i)
        coeff = isodd(i) ? T(4) : T(2)
        acc += coeff * f(x)
    end

    return acc * (h / T(3))
end

function Lp(f::Function, p::Real, a::T, b::T; h::T=T(1e-4)) where {T<:Real}
    Tp = T(p)
    return integral(x -> abs(f(x))^Tp, a, b; h=h)^(1 / Tp)
end

# === Trapz ===

"""
    trapz(
        x::AbstractVector{S},
        y::AbstractVector{T},
        linear::Bool = false
    ) where {S<:Number, T<:Number}

Perform numerical integration by using the composite trapezoidal rule.
Note that it supports arbitrary precision via BigFloat.

### Arguments
* x      -> Real frequency mesh.
* y      -> Function values at real axis.
* linear -> Whether the given mesh is linear?

### Returns
* ℐ -> The final value.

See also: [`simpson`](@ref).
"""
function trapz(x::AbstractVector{S},
               y::AbstractVector{T},
               linear::Bool=false) where {S<:Number,T<:Number}
    # For linear mesh
    if linear
        h = x[2] - x[1]
        value = y[1] + y[end] + 2 * sum(y[2:(end - 1)])
        value = h * value / 2
        # For non-equidistant mesh
    else
        len = length(x)
        dx = view(x, 2:len) .- view(x, 1:(len - 1))
        y_forward = view(y, 2:len)
        y_backward = view(y, 1:(len - 1))
        value = sum((1 // 2) * (y_forward .+ y_backward) .* dx)
    end

    return value
end

# === Simpson ===

"""
    simpson(
        x::AbstractVector{S},
        y::AbstractVector{T}
    ) where {S<:Number, T<:Number}

Perform numerical integration by using the simpson rule. And `x` must be a linear and
uniform mesh.

### Arguments
* x -> Real frequency mesh.
* y -> Function values at real axis.

### Returns
* ℐ -> The final value.

See also: [`trapz`](@ref).
"""
function simpson(x::AbstractVector{S},
                 y::AbstractVector{T}) where {S<:Number,T<:Number}
    n = length(x)
    @assert n == length(y)

    if isodd(n)
        even_sum = sum(view(y, 2:2:(n - 1)))
        odd_sum = sum(view(y, 3:2:(n - 1)))
        return (y[1] + y[end] + 4 * even_sum + 2 * odd_sum) * (x[2] - x[1]) / 3
    else
        even_sum = sum(view(y, 2:2:(n - 4)))
        odd_sum = sum(view(y, 3:2:(n - 4)))
        res = (y[1] + y[end - 3] + 4 * even_sum + 2 * odd_sum) * 1 // 3
        res += 3 // 8 * (y[end - 3] + y[end]) + 9 // 8 * (y[end - 2] + y[end - 1])
        return res * (x[2] - x[1])
    end
end
