tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

# loss function
function loss(G::Vector{T}, G₀::Vector{T}, w::Vector{T}) where {T<:Real}
    return sqrt(sum(abs2.(G .- G₀) .* w))
end

# default configuration
function dfcfg_cont(T::Type{<:Real};
                    μ=[T(1 // 2), T(-5 // 2)]::Vector{T},
                    σ=[T(1 // 5), T(4 // 5)]::Vector{T},
                    amplitude=[T(1), T(3 // 10)]::Vector{T},
                    mesh_type=UniformMesh(),
                    β=T(10)::T,
                    N=20,
                    noise=T(0)::T,
                    mb=T(8)::T,
                    ml=801::Int,)
    A=continous_spectral_density(μ, σ, amplitude)
    ctx = CtxData(β, N; mesh_bound=mb, mesh_length=ml, mesh_type=mesh_type)
    GFV = generate_GFV_cont(β, N, A; noise=noise)
    return A, ctx, GFV
end

function jacobian_check_m2v(f, J::Matrix{T}, A::Matrix{T}; η=1e-5, rtol=1e-2,
                            atol=1e-8) where {T<:Number}
    η = T(η)
    dA = η * A/norm(A)
    dy = f(A+dA) - f(A)
    dy_expect = real(conj(J)*vec(dA))
    return isapprox(dy, dy_expect; rtol=rtol, atol=atol)
end
function jacobian_check_v2v(f, J::Matrix{T}, x::Vector{T}; η=1e-5, rtol=1e-2,
                            atol=1e-8) where {T<:Number}
    η = T(η)
    dx = η * x/norm(x)
    dy = f(x+dx) - f(x)
    dy_expect = real(conj(J)*dx)
    return isapprox(dy, dy_expect; rtol=rtol, atol=atol)
end
