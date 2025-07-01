tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

# loss function
function loss(G::Vector{T}, G₀::Vector{T}, w::Vector{S}) where {T<:Number,S<:Real}
    @assert S == real(T)
    return sqrt(sum(abs2.(G .- G₀) .* w))
end

# default configuration
function dfcfg(T::Type{<:Real};
               μ=[T(1 // 2), T(-5 // 2)]::Vector{T},
               σ=[T(1 // 5), T(4 // 5)]::Vector{T},
               amplitudes=[T(1), T(3 // 10)]::Vector{T},
               mesh_type=UniformMesh(),
               β=T(10)::T,
               N=20,
               GFVσ=T(1e-4)::T,
               noise=T(0)::T,
               mb=T(8)::T,
               ml=801::Int,
               spt::SpectrumType=Cont(),
               poles_num::Int=4,)
    ctx = CtxData(β, N; mesh_bound=mb, mesh_length=ml, mesh_type=mesh_type, σ=GFVσ)
    if spt isa Cont
        A=continous_spectral_density(μ, σ, amplitudes)
        GFV = generate_GFV_cont(β, N, A; noise=noise)
        return A, ctx, GFV
    elseif spt isa Delta
        poles = collect(1:poles_num) .+ rand(T, poles_num) * T(1//2)
        γ = rand(T, poles_num)
        γ ./= sum(γ)
        GFV = generate_GFV_delta(β, N, poles, γ; noise=noise)
        return (poles, γ), ctx, GFV
    end
end
#=
function jacobian_check_m2v(f, J::Matrix{T}, A::Matrix{T}; η=1e-5, rtol=1e-2,
                            atol=1e-8) where {T<:Number}
    η = T(η)
    dA = η * A/norm(A)
    dy = f(A+dA) - f(A)
    dy_expect = real(conj(J)*vec(dA))
    return isapprox(dy, dy_expect; rtol=rtol, atol=atol)
end
=#
# vector to vector
function jacobian_check_v2v(f, J::Matrix{T}, x::Vector{T}; η=1e-5, rtol=1e-2,
                            atol=1e-8) where {T<:Number}
    η = T(η)
    dx = η * x/norm(x)
    dy = f(x+dx) - f(x)
    dy_expect = real(conj(J)*dx)
    err = norm(dy-dy_expect)
    rel_err = err/min(norm(dy), norm(dy_expect))
    @show err
    @show rel_err
    return isapprox(dy, dy_expect; rtol=rtol, atol=atol)
end
# vector to number
function gradient_check(f, J::Vector{T}, x::Vector{T}; η=1e-5, rtol=1e-2,
                        atol=tolerance(T)) where {T<:Number}
    η = real(T(η))
    #η = min(η0, findmin(abs.(x))[1])
    #atol = atol / η0 * η
    dx = η * J/norm(J)
    dy = f(x+dx) - f(x)
    dy_expect = η * norm(J)
    @show dy
    @show dy_expect
    return isapprox(dy, dy_expect; rtol=rtol, atol=atol)
end

function ave_grad(J::Matrix{T}; try_num::Int=1000) where {T<:Number}
    res = zeros(real(T), size(J, 1))
    N = size(J, 2)
    for i in 1:try_num
        dx = randn(T, N)
        res += abs.(real(conj(J)*dx))
    end
    return res / try_num
end
