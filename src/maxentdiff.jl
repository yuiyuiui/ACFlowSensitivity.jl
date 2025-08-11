#TODO: 1. add diff for blur; 2. add diff for offdiag; 3. consider vector σ

# function χ²
struct funcχ²{T<:Real}<:Function
    KΔ::Matrix{T}
    σ⁻²::T
end
function funcχ²(mec::MaxEntContext{T}) where {T<:Real}
    return funcχ²(mec.kernel * Diagonal(mec.δ), mec.σ⁻²[1])
end
function (f::funcχ²{T})(A::Vector{T}, G::Vector{T}) where {T<:Real}
    return norm((f.KΔ * A - G))^2 * f.σ⁻²
end

# function ∇Aχ²
struct func∇Aχ²{T<:Real}<:Function
    KΔ::Matrix{T}
    σ⁻²::T
end
function func∇Aχ²(mec::MaxEntContext{T}) where {T<:Real}
    return func∇Aχ²(mec.kernel * Diagonal(mec.δ), mec.σ⁻²[1])
end
function (f::func∇Aχ²{T})(A::Vector{T}, G::Vector{T}) where {T<:Real}
    return 2 * f.σ⁻² * f.KΔ' * (f.KΔ * A - G)
end

# function ∇Gχ²
struct func∇Gχ²{T<:Real}<:Function
    KΔ::Matrix{T}
    σ⁻²::T
end
function func∇Gχ²(mec::MaxEntContext{T}) where {T<:Real}
    return func∇Gχ²(mec.kernel * Diagonal(mec.δ), mec.σ⁻²[1])
end
function (f::func∇Gχ²{T})(A::Vector{T}, G::Vector{T}) where {T<:Real}
    return 2 * f.σ⁻² * (G - f.KΔ * A)
end

# function A, including Asj and Abr
abstract type funcA{T<:Real}<:Function end
# function Asj
struct funcAsj{T<:Real}<:funcA{T}
    model::Vector{T}
    V::Matrix{T}
end
function funcAsj(mec::MaxEntContext{T}) where {T<:Real}
    return funcAsj(mec.model, mec.V)
end
function (f::funcAsj{T})(u::Vector{T}) where {T<:Real}
    return f.model .* exp.(f.V * u)
end

# function Abr
struct funcAbr{T<:Real}<:funcA{T}
    model::Vector{T}
    V::Matrix{T}
end
function funcAbr(mec::MaxEntContext{T}) where {T<:Real}
    return funcAbr(mec.model, mec.V)
end
function (f::funcAbr{T})(u::Vector{T}) where {T<:Real}
    return f.model ./ (1 .- f.model .* f.V * u)
end

# function A
function funcA(mec::MaxEntContext{T}) where {T<:Real}
    if mec.stype isa SJ
        return funcAsj(mec)
    elseif mec.stype isa BR
        return funcAbr(mec)
    else
        error("Unsupported spectral function type")
    end
end

# function ∂A/∂u, including ∂Asj/∂u and ∂Abr/∂u
abstract type func∂ADiv∂u{T<:Real}<:Function end

# function ∂Asj/∂u
struct func∂AsjDiv∂u{T<:Real}<:func∂ADiv∂u{T}
    V::Matrix{T}
    model::Vector{T}
end
function func∂AsjDiv∂u(mec::MaxEntContext{T}) where {T<:Real}
    return func∂AsjDiv∂u(mec.V, mec.model)
end
function (f::func∂AsjDiv∂u{T})(u::Vector{T}) where {T<:Real}
    return Diagonal(f.model .* exp.(f.V * u)) * f.V
end

# function ∇uAbr
struct func∂AbrDiv∂u{T<:Real}<:func∂ADiv∂u{T}
    V::Matrix{T}
    model::Vector{T}
end
function func∂AbrDiv∂u(mec::MaxEntContext{T}) where {T<:Real}
    return func∂AbrDiv∂u(mec.V, mec.model)
end
function (f::func∂AbrDiv∂u{T})(u::Vector{T}) where {T<:Real}
    return Diagonal((f.model ./ (1 .- f.model .* f.V * u)) .^ 2) * f.V
end

# function ∂A/∂u
function func∂ADiv∂u(mec::MaxEntContext{T}) where {T<:Real}
    if mec.stype isa SJ
        return func∂AsjDiv∂u(mec)
    elseif mec.stype isa BR
        return func∂AbrDiv∂u(mec)
    else
        error("Unsupported spectral function type")
    end
end

# function S, including Ssj, Sbr to add

# function ∇AS
struct func∇AS{T<:Real}<:Function
    ΔV::Matrix{T}
end
function func∇AS(mec::MaxEntContext{T}) where {T<:Real}
    return func∇AS(Diagonal(mec.δ) * mec.V)
end
function (f::func∇AS{T})(u::Vector{T}) where {T<:Real}
    return -f.ΔV * u
end

#= function Q
struct funcQ{T<:Real}<:Function
    χ²::funcχ²{T}  # χ²: A, G -> χ²(A, G)
    S::funcS{T}  # S: A -> S(A)
end
function funcQ(mec::MaxEntContext{T}) where {T<:Real}
    return funcQ(funcχ²(mec), funcS(mec))
end
function (f::funcQ{T})(A::Vector{T}, α::T, G::Vector{T}) where {T<:Real}
    return -f.χ²(A, G)/2 + α * f.S(A)
end

# ∇GQ = -∇Gχ²/2

# ∇AQ = α∇AS - ∇Gχ²/2
struct func∇AQ{T<:Real}<:Function
    ∇AS::func∇AS{T}
    ∇Gχ²::func∇Gχ²{T}
end
function func∇AQ(mec::MaxEntContext{T}) where {T<:Real}
    return func∇AQ(func∇AS(mec), func∇Gχ²(mec))
end
function (f::func∇AQ{T})(A::Vector{T}, α::T, G::Vector{T}) where {T<:Real}
    return α * f.∇AS(A) - f.∇Gχ²(A, G) / 2
end
=#

# function ƒ
struct funcƒ{T<:Real}<:Function
    ΣUᵗσ⁻²::Matrix{T}
    KΔ::Matrix{T}
    A::funcA{T}  # A: u -> A(u)
end
function funcƒ(mec::MaxEntContext{T}) where {T<:Real}
    return funcƒ(mec.Σ * mec.U' * mec.σ⁻²[1], mec.kernel * Diagonal(mec.δ), funcA(mec))
end
function (f::funcƒ{T})(u::Vector{T}, α::T, G::Vector{T}) where {T<:Real}
    return f.ΣUᵗσ⁻² * (f.KΔ * f.A(u) - G) + α * u
end

# function J
struct funcJ{T<:Real}<:Function
    Σ²VᵗΔσ⁻²::Matrix{T}
    ∂ADiv∂u::func∂ADiv∂u{T}
end
function funcJ(mec::MaxEntContext{T}) where {T<:Real}
    return funcJ(mec.Σ^2 * mec.V' * Diagonal(mec.δ) * mec.σ⁻²[1], func∂ADiv∂u(mec))
end
function (f::funcJ{T})(u::Vector{T}, α::T) where {T<:Real}
    return f.Σ²VᵗΔσ⁻² * f.∂ADiv∂u(u) + α * I(length(u))
end

# function ∂ƒ/∂α
∂ƒDiv∂α(u::Vector{T}) where {T<:Real} = u

# function ∂ƒ/∂G
struct func∂ƒDiv∂G{T<:Real}<:Function
    ΣUᵗσ⁻²::Matrix{T}
end
function func∂ƒDiv∂G(mec::MaxEntContext{T}) where {T<:Real}
    return func∂ƒDiv∂G(mec.Σ * mec.U' * mec.σ⁻²[1])
end
function (f::func∂ƒDiv∂G{T})() where {T<:Real}
    return -f.ΣUᵗσ⁻²
end

struct func∂uαDiv∂G{T<:Real}<:Function
    J::funcJ{T}
    ∂ƒDiv∂G::func∂ƒDiv∂G{T}
end
function func∂uαDiv∂G(mec::MaxEntContext{T}) where {T<:Real}
    return func∂uαDiv∂G(funcJ(mec), func∂ƒDiv∂G(mec))
end
function (f::func∂uαDiv∂G{T})(u::Vector{T}, α::T) where {T<:Real}
    return - pinv(f.J(u, α)) * f.∂ƒDiv∂G()
end

# ============ solvediff ===================
function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::MaxEnt) where {T<:Real}
    alg.offdiag && error("offdiag is not supported for maxent diff now")

    ctx.spt isa Delta && return pγdiff(GFV, ctx, alg; equalγ=false)
    !(ctx.spt isa Cont) && error("Maxent now only support continuous and delta spectrum")

    mec = init(GFV, ctx, alg)

    if alg.method == "chi2kink"
        return chi2kinkdiff(mec, alg)
    elseif alg.method == "bryan"
        return bryandiff(mec, alg)
    elseif alg.method == "historic"
        return historicdiff(mec, alg)
    elseif alg.method == "classic"
        return classicdiff(mec, alg)
    else
        error("Unsupported method")
    end
end

# ============ chi2kink_diff ===================
function chi2kinkdiff(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    function fitfun(x, p)
        return @. p[1] + p[2] / (T(1) + exp(-p[4] * (x - p[3])))
    end
    println("Apply chi2kink algorithm to determine optimized α")

    use_bayes = false
    alpha = T(alg.alpha)
    ratio = T(alg.ratio)
    nalph = alg.nalph
    n_svd = length(mec.Bₘ)
    G = mec.Gᵥ
    uαvec = Vector{T}[]
    u₀ = zeros(T, n_svd)
    χ²vec = T[]
    αvec = T[]

    for i in 1:nalph
        sol = optimizer(mec, alpha/ratio^(i-1), u₀, use_bayes, alg)
        push!(αvec, alpha/ratio^(i-1))
        push!(χ²vec, sol[:χ²])
        @. u₀ = sol[:u]
        push!(uαvec, sol[:u])
    end

    good = isfinite.(χ²vec)
    guess = [T(0), T(5), T(2), T(0)]
    fit = curve_fit(fitfun, log10.(αvec[good]), log10.(χ²vec[good]), guess)
    p = fit.param
    _, _, c, d = p

    # `fit_pos` is a control parameter for under/overfitting.
    # Good values are usually between 2 and 2.5. Smaller values usually
    # lead to underfitting, which is sometimes desirable. Larger values
    # lead to overfitting, which should be avoided.
    fit_pos = T(2.5)
    αopt = c - fit_pos / d
    close = argmin(abs.(log10.(αvec) .- αopt))
    u₀ = uαvec[close]
    αopt = T(10) ^ αopt

    sol = optimizer(mec, αopt, u₀, use_bayes, alg)
    uopt = sol[:u]
    println("Optimized α : $αopt log10(α) : $(log10(αopt))")

    _A = funcA(mec)
    _∇Aχ² = func∇Aχ²(mec)
    _∇Gχ² = func∇Gχ²(mec)
    _∂ADiv∂u = func∂ADiv∂u(mec)
    _∂uαDiv∂G = func∂uαDiv∂G(mec)
    _J = funcJ(mec)
    _∂ƒDiv∂G = func∂ƒDiv∂G(mec)
    ∂χ²vecDiv∂G = zeros(T, nalph, length(G))
    for i in 1:nalph
        if good[i]
            ∂χ²vecDiv∂G[i, :] = _∇Aχ²(_A(uαvec[i]), G)' * _∂ADiv∂u(uαvec[i]) *
                                _∂uαDiv∂G(uαvec[i], αvec[i]) + _∇Gχ²(_A(uαvec[i]), G)'
        end
    end

    ∂pDiv∂χ²vec = -pinv(∂²lossϕDiv∂p²(p, log10.(αvec), log10.(χ²vec))) *
                  ∂²lossϕDiv∂p∂y(p, log10.(αvec), log10.(χ²vec)) *
                  Diagonal(1 ./ (χ²vec * T(log(10))))

    ∂αoptDiv∂p = [T(0) T(0) αopt * T(log(10)) αopt * T(log(10)) * fit_pos / d^2]
    ∂αoptDiv∂G = ∂αoptDiv∂p * ∂pDiv∂χ²vec * ∂χ²vecDiv∂G
    ∂uoptDiv∂G = - pinv(_J(uopt, αopt)) * (_∂ƒDiv∂G() + uopt * ∂αoptDiv∂G)
    ∂AoptDiv∂G = _∂ADiv∂u(uopt) * ∂uoptDiv∂G

    N = size(mec.kernel, 1) ÷ 2
    return _A(uopt), ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)], alg.test
end
