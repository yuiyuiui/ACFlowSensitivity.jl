#TODO: 1. add diff for blur; 2. add diff for offdiag; 3. consider vector σ

# function χ²
struct funcχ²{T<:Real} <: Function
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
struct func∇Aχ²{T<:Real} <: Function
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
struct func∇Gχ²{T<:Real} <: Function
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
abstract type funcA{T<:Real} <: Function end
# function Asj
struct funcAsj{T<:Real} <: funcA{T}
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
struct funcAbr{T<:Real} <: funcA{T}
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
abstract type func∂ADiv∂u{T<:Real} <: Function end

# function ∂Asj/∂u
struct func∂AsjDiv∂u{T<:Real} <: func∂ADiv∂u{T}
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
struct func∂AbrDiv∂u{T<:Real} <: func∂ADiv∂u{T}
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
struct func∇AS{T<:Real} <: Function
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
struct funcƒ{T<:Real} <: Function
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
struct funcJ{T<:Real} <: Function
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
struct func∂ƒDiv∂G{T<:Real} <: Function
    ΣUᵗσ⁻²::Matrix{T}
end
function func∂ƒDiv∂G(mec::MaxEntContext{T}) where {T<:Real}
    return func∂ƒDiv∂G(mec.Σ * mec.U' * mec.σ⁻²[1])
end
function (f::func∂ƒDiv∂G{T})() where {T<:Real}
    return -f.ΣUᵗσ⁻²
end

struct func∂uαDiv∂G{T<:Real} <: Function
    J::funcJ{T}
    ∂ƒDiv∂G::func∂ƒDiv∂G{T}
end
function func∂uαDiv∂G(mec::MaxEntContext{T}) where {T<:Real}
    return func∂uαDiv∂G(funcJ(mec), func∂ƒDiv∂G(mec))
end
function (f::func∂uαDiv∂G{T})(u::Vector{T}, α::T) where {T<:Real}
    return -pinv(f.J(u, α)) * f.∂ƒDiv∂G()
end

struct func∂uαDiv∂α{T<:Real} <: Function
    J::funcJ{T}
end
function func∂uαDiv∂α(mec::MaxEntContext{T}) where {T<:Real}
    return func∂uαDiv∂α(funcJ(mec))
end
function (f::func∂uαDiv∂α{T})(u::Vector{T}, α::T) where {T<:Real}
    return -pinv(f.J(u, α)) * u
end

# ============ solvediff ===================
function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::MaxEnt) where {T<:Real}
    alg.offdiag && error("offdiag is not supported for maxent diff now")
    alg.stype isa BR && ctx.spt isa Cont &&
        error("It's recommended to use BR for Delta spectrum")

    ctx.spt isa Delta && return pγdiff(GFV, ctx, alg)
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
    Avec = Vector{T}[]
    u₀ = zeros(T, n_svd)
    χ²vec = T[]
    αvec = T[]

    for i in 1:nalph
        sol = optimizer(mec, alpha / ratio^(i - 1), u₀, use_bayes, alg)
        push!(αvec, alpha / ratio^(i - 1))
        push!(χ²vec, sol[:χ²])
        @. u₀ = sol[:u]
        push!(uαvec, sol[:u])
        push!(Avec, sol[:A])
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
    αopt = T(10)^αopt

    sol = optimizer(mec, αopt, u₀, use_bayes, alg)
    uopt = sol[:u]
    Aopt = sol[:A]
    println("Optimized α : $αopt log10(α) : $(log10(αopt))")

    _∇Aχ² = func∇Aχ²(mec)
    _∇Gχ² = func∇Gχ²(mec)
    _∂ADiv∂u = func∂ADiv∂u(mec)
    _∂uαDiv∂G = func∂uαDiv∂G(mec) # u,α ->
    _J = funcJ(mec)
    _∂ƒDiv∂G = func∂ƒDiv∂G(mec)
    ∂χ²vecDiv∂G = zeros(T, nalph, length(G))
    for i in 1:nalph
        if good[i]
            ∂χ²vecDiv∂G[i, :] = _∇Aχ²(Avec[i], G)' * _∂ADiv∂u(uαvec[i]) *
                                _∂uαDiv∂G(uαvec[i], αvec[i]) + _∇Gχ²(Avec[i], G)'
        end
    end

    ∂pDiv∂χ²vec = -pinv(∂²lossϕDiv∂p²(p, log10.(αvec), log10.(χ²vec))) *
                  ∂²lossϕDiv∂p∂y(p, log10.(αvec), log10.(χ²vec)) *
                  Diagonal(1 ./ (χ²vec * T(log(10))))

    ∂αoptDiv∂p = [T(0) T(0) αopt * T(log(10)) αopt * T(log(10)) * fit_pos / d^2]
    ∂αoptDiv∂G = ∂αoptDiv∂p * ∂pDiv∂χ²vec * ∂χ²vecDiv∂G
    ∂uoptDiv∂G = -pinv(_J(uopt, αopt)) * (_∂ƒDiv∂G() + uopt * ∂αoptDiv∂G)
    ∂AoptDiv∂G = _∂ADiv∂u(uopt) * ∂uoptDiv∂G

    N = size(mec.kernel, 1) ÷ 2
    return Aopt, ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

# ============= bryan ===================
function bryandiff(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    G = mec.Gᵥ
    m = length(mec.δ)
    solvec, sol = bryan(mec, alg)
    nalph = length(solvec)
    αvec = [solvec[i][:α] for i in 1:nalph]
    ∂PαvecDiv∂G = zeros(T, nalph, length(G))
    _∇Aχ² = func∇Aχ²(mec) # A,G ->
    _∇AS = func∇AS(mec) # u ->
    _∇Gχ² = func∇Gχ²(mec) # A,G ->
    _∂uαDiv∂G = func∂uαDiv∂G(mec)
    nsvd = length(solvec[1][:u])
    ∂uαvecDiv∂G = zeros(T, nalph * nsvd, length(G))
    ∇AQ = [-_∇Aχ²(solvec[i][:A], G) / 2 + _∇AS(solvec[i][:u]) * solvec[i][:α]
           for i in 1:nalph]

    _T = v -> sqrt.(v ./ mec.δ)
    _∂TDiv∂A = v -> Diagonal(1 ./ sqrt.(v .* mec.δ))

    H = Diagonal(mec.δ) * mec.kernel' * mec.kernel * Diagonal(mec.δ)
    Pvec = T[]
    Aαvec = Vector{T}[]

    for i in 1:nalph
        uᵢ = solvec[i][:u]
        Aᵢ = solvec[i][:A]
        push!(Aαvec, Aᵢ)
        Tᵢ = _T(Aᵢ)
        Pᵢ = solvec[i][:prob]
        push!(Pvec, Pᵢ)
        αᵢ = solvec[i][:α]
        Λᵢ⁻¹ = invΛ(αᵢ, Diagonal(Tᵢ) * H * Diagonal(Tᵢ))
        ∇APᵢ = Pᵢ * (∇AQ[i] - _∂TDiv∂A(Aᵢ) * diag(Λᵢ⁻¹ * Diagonal(Tᵢ) * H))
        ∇GPᵢ = Pᵢ * _∇Gχ²(Aᵢ, G)
        ∂uαDiv∂Gᵢ = _∂uαDiv∂G(uᵢ, αᵢ)
        ∂ADiv∂uᵢ = Diagonal(Aᵢ) * mec.V

        ∂PαvecDiv∂G[i, :] = ∇GPᵢ' + ∇APᵢ' * ∂ADiv∂uᵢ * ∂uαDiv∂Gᵢ
        ∂uαvecDiv∂G[((i - 1) * nsvd + 1):(i * nsvd), :] .= ∂uαDiv∂Gᵢ
    end

    function last(Avec, Pᵥ)
        Pᵥnorm = -Pᵥ ./ trapz(αvec, Pᵥ)
        spectra = hcat([Avec[((i - 1) * m + 1):(i * m)] * Pᵥnorm[i] for i in 1:nalph]...)
        Aopt = [-trapz(αvec, spectra[j, :]) for j in 1:m]
        return Aopt
    end

    Aout = last(vcat(Aαvec...), Pvec)
    ∂AsumDiv∂Avec, ∂AsumDiv∂Pᵥ = Zygote.jacobian(last, vcat(Aαvec...), Pvec)
    ∂AsumDiv∂uvec = zeros(T, m, nsvd * nalph)
    for i in 1:nalph
        ∂AsumDiv∂uvec[:, ((i - 1) * nsvd + 1):(i * nsvd)] .= ∂AsumDiv∂Avec[:,
                                                                           ((i - 1) * m + 1):(i * m)] *
                                                             Diagonal(Aαvec[i]) * mec.V
    end

    ∂AoptDiv∂G = ∂AsumDiv∂uvec * ∂uαvecDiv∂G + ∂AsumDiv∂Pᵥ * ∂PαvecDiv∂G
    N = length(G) ÷ 2
    return Aout, ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

# Λ⁻¹ =  (αI + UΣU')⁻¹ = I/α - U(α²Σ⁻¹ + αI)⁻¹ U'
function invΛ(α::T, H::Matrix{T}) where {T<:Real}
    S, U = eigen(Hermitian(H))
    idx = findall(S .> strict_tol(T))
    S = S[idx]
    U = U[:, idx]
    m, n = size(U)
    return I(m) / α - U * Diagonal((α^2 ./ S + α * ones(T, n)) .^ (-1)) * U'
end

# ============= historic ===================
function historicdiff(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    _, sol = historic(mec, alg)
    ∇Aχ² = func∇Aχ²(mec)(sol[:A], mec.Gᵥ)

    ∂ADiv∂u = Diagonal(sol[:A]) * mec.V

    ∂uαDiv∂α = func∂uαDiv∂α(mec)(sol[:u], sol[:α])
    ∇Gχ² = func∇Gχ²(mec)(sol[:A], mec.Gᵥ)
    ∂uαDiv∂G = func∂uαDiv∂G(mec)(sol[:u], sol[:α])
    tmp = ∇Aχ²' * ∂ADiv∂u
    ∂αoptDiv∂G = -(∇Gχ²' + tmp * ∂uαDiv∂G) / (tmp * ∂uαDiv∂α)
    ∂uoptDiv∂G = ∂uαDiv∂G + ∂uαDiv∂α * ∂αoptDiv∂G
    ∂AoptDiv∂G = ∂ADiv∂u * ∂uoptDiv∂G
    N = length(mec.Gᵥ) ÷ 2
    return sol[:A], ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

# ============= classic ===================
function classicdiff(mec::MaxEntContext{R}, alg::MaxEnt) where {R<:Real}
    _, sol = classic(mec, alg)
    α = sol[:α]
    A = sol[:A]
    u = sol[:u]
    S = sol[:S]
    T = sqrt.(A ./ mec.δ)
    ∂TDiv∂A = Diagonal(1 ./ (2 * sqrt.(A ./ mec.δ)))
    ∂ADiv∂u = Diagonal(A) * mec.V

    ∂SDiv∂A = -(Diagonal(mec.δ) * mec.V * u)'
    # construct Λ
    H = Diagonal(mec.δ) * mec.kernel' * mec.kernel * Diagonal(mec.δ)
    Λ = H * Diagonal(T) * invΛ(α, Diagonal(T) * H * Diagonal(T))^2

    ∂uαDiv∂α = func∂uαDiv∂α(mec)(u, α)
    ∂uαDiv∂G = func∂uαDiv∂G(mec)(u, α)
    tmp = (diag(Λ)' * ∂TDiv∂A + ∂SDiv∂A) * ∂ADiv∂u
    ∂αoptDiv∂G = -α / (S + α * tmp * ∂uαDiv∂α) * tmp * ∂uαDiv∂G
    ∂uoptDiv∂G = ∂uαDiv∂G + ∂uαDiv∂α * ∂αoptDiv∂G
    ∂AoptDiv∂G = ∂ADiv∂u * ∂uoptDiv∂G
    N = length(mec.Gᵥ) ÷ 2
    return A, ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

#= use BR for Delta spectrum

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
    Avec = Vector{T}[]
    u₀ = zeros(T, n_svd)
    χ²vec = T[]
    αvec = T[]

    for i in 1:nalph
        sol = optimizer(mec, alpha/ratio^(i-1), u₀, use_bayes, alg)
        push!(αvec, alpha/ratio^(i-1))
        push!(χ²vec, sol[:χ²])
        @. u₀ = sol[:u]
        push!(uαvec, sol[:u])
        push!(Avec, sol[:A])
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
    Aopt = sol[:A]
    println("Optimized α : $αopt log10(α) : $(log10(αopt))")

    _∇Aχ² = func∇Aχ²(mec)
    _∇Gχ² = func∇Gχ²(mec)
    _∂ADiv∂u = func∂ADiv∂u(mec)
    _∂uαDiv∂G = func∂uαDiv∂G(mec) # u,α ->
    _J = funcJ(mec)
    _∂ƒDiv∂G = func∂ƒDiv∂G(mec)
    ∂χ²vecDiv∂G = zeros(T, nalph, length(G))
    for i in 1:nalph
        if good[i]
            ∂χ²vecDiv∂G[i, :] = _∇Aχ²(Avec[i], G)' * _∂ADiv∂u(uαvec[i]) *
                                _∂uαDiv∂G(uαvec[i], αvec[i]) + _∇Gχ²(Avec[i], G)'
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
    return Aopt, ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

# ============= bryan ===================
function bryandiff(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    G = mec.Gᵥ
    m = length(mec.δ)
    solvec, sol = bryan(mec, alg)
    nalph = length(solvec)
    αvec = [solvec[i][:α] for i in 1:nalph]
    ∂PαvecDiv∂G = zeros(T, nalph, length(G))
    _∇Aχ² = func∇Aχ²(mec) # A,G ->
    _∇AS = func∇AS(mec) # u ->
    _∇Gχ² = func∇Gχ²(mec) # A,G ->
    _∂uαDiv∂G = func∂uαDiv∂G(mec)
    nsvd = length(solvec[1][:u])
    ∂uαvecDiv∂G = zeros(T, nalph * nsvd, length(G))
    ∇AQ = [- _∇Aχ²(solvec[i][:A], G) / 2 + _∇AS(solvec[i][:u]) * solvec[i][:α]
           for i in 1:nalph]
    if mec.stype isa SJ
        _T = v -> sqrt.(v ./ mec.δ)
        _∂TDiv∂A = v -> Diagonal(1 ./ sqrt.(v .* mec.δ))
    elseif mec.stype isa BR
        _T = v -> v ./ sqrt.(mec.δ)
        _∂TDiv∂A = v -> Diagonal(1 ./ sqrt.(mec.δ))
    else
        error("Unsupported entropy type")
    end
    H = Diagonal(mec.δ) * mec.kernel' * mec.kernel * Diagonal(mec.δ)
    Pvec = T[]
    Aαvec = Vector{T}[]

    for i in 1:nalph
        uᵢ = solvec[i][:u]
        Aᵢ = solvec[i][:A]
        push!(Aαvec, Aᵢ)
        Tᵢ = _T(Aᵢ)
        Pᵢ = solvec[i][:prob]
        push!(Pvec, Pᵢ)
        αᵢ = solvec[i][:α]
        Λᵢ⁻¹ = invΛ(αᵢ, Diagonal(Tᵢ) * H * Diagonal(Tᵢ))
        ∇APᵢ = Pᵢ * (∇AQ[i] - _∂TDiv∂A(Aᵢ)*diag(Λᵢ⁻¹ * Diagonal(Tᵢ) * H))
        ∇GPᵢ = Pᵢ * _∇Gχ²(Aᵢ, G)
        ∂uαDiv∂Gᵢ = _∂uαDiv∂G(uᵢ, αᵢ)
        if alg.stype isa SJ
            ∂ADiv∂uᵢ = Diagonal(Aᵢ) * mec.V
        elseif alg.stype isa BR
            ∂ADiv∂uᵢ = Diagonal(Aᵢ .^ 2) * mec.V
        else
            error("Unsupported entropy type")
        end
        ∂PαvecDiv∂G[i, :] = ∇GPᵢ' + ∇APᵢ' * ∂ADiv∂uᵢ * ∂uαDiv∂Gᵢ
        ∂uαvecDiv∂G[((i - 1) * nsvd + 1):(i * nsvd), :] .= ∂uαDiv∂Gᵢ
    end

    function last(Avec, Pᵥ)
        Pᵥnorm = -Pᵥ ./ trapz(αvec, Pᵥ)
        spectra = hcat([Avec[((i - 1) * m + 1):(i * m)] * Pᵥnorm[i] for i in 1:nalph]...)
        Aopt = [-trapz(αvec, spectra[j, :]) for j in 1:m]
        return Aopt
    end

    Aout = last(vcat(Aαvec...), Pvec)
    ∂AsumDiv∂Avec, ∂AsumDiv∂Pᵥ = Zygote.jacobian(last, vcat(Aαvec...), Pvec)
    ∂AsumDiv∂uvec = zeros(T, m, nsvd * nalph)
    for i in 1:nalph
        if alg.stype isa SJ
            ∂AsumDiv∂uvec[:, ((i - 1) * nsvd + 1):(i * nsvd)] .= ∂AsumDiv∂Avec[:,
                                                                               ((i - 1) * m + 1):(i * m)] *
                                                                 Diagonal(Aαvec[i]) * mec.V
        elseif alg.stype isa BR
            ∂AsumDiv∂uvec[:, ((i - 1) * nsvd + 1):(i * nsvd)] .= ∂AsumDiv∂Avec[:,
                                                                               ((i - 1) * m + 1):(i * m)] *
                                                                 Diagonal(Aαvec[i] .^ 2) *
                                                                 mec.V
        else
            error("Unsupported entropy type")
        end
    end

    ∂AoptDiv∂G = ∂AsumDiv∂uvec * ∂uαvecDiv∂G + ∂AsumDiv∂Pᵥ * ∂PαvecDiv∂G
    N = length(G) ÷ 2
    return Aout, ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

# Λ⁻¹ =  (αI + UΣU')⁻¹ = I/α - U(α²Σ⁻¹ + αI)⁻¹ U'
function invΛ(α::T, H::Matrix{T}) where {T<:Real}
    S, U = eigen(Hermitian(H))
    idx = findall(S .> strict_tol(T))
    S = S[idx]
    U = U[:, idx]
    m, n = size(U)
    return I(m)/α - U * Diagonal((α^2 ./ S + α * ones(T, n)) .^ (-1)) * U'
end

# ============= historic ===================
function historicdiff(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    _, sol = historic(mec, alg)
    ∇Aχ² = func∇Aχ²(mec)(sol[:A], mec.Gᵥ)
    if mec.stype isa SJ
        ∂ADiv∂u = Diagonal(sol[:A]) * mec.V
    elseif alg.stype isa BR
        ∂ADiv∂u = Diagonal(sol[:A] .^ 2) * mec.V
    else
        error("Unsupported entropy type")
    end
    ∂uαDiv∂α = func∂uαDiv∂α(mec)(sol[:u], sol[:α])
    ∇Gχ² = func∇Gχ²(mec)(sol[:A], mec.Gᵥ)
    ∂uαDiv∂G = func∂uαDiv∂G(mec)(sol[:u], sol[:α])
    tmp = ∇Aχ²' * ∂ADiv∂u
    ∂αoptDiv∂G = -(∇Gχ²' + tmp * ∂uαDiv∂G) / (tmp * ∂uαDiv∂α)
    ∂uoptDiv∂G = ∂uαDiv∂G + ∂uαDiv∂α * ∂αoptDiv∂G
    ∂AoptDiv∂G = ∂ADiv∂u * ∂uoptDiv∂G
    N = length(mec.Gᵥ) ÷ 2
    return sol[:A], ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

# ============= classic ===================
function classicdiff(mec::MaxEntContext{R}, alg::MaxEnt) where {R<:Real}
    _, sol = classic(mec, alg)
    @show 1
    α = sol[:α]
    A = sol[:A]
    u = sol[:u]
    S = sol[:S]
    if mec.stype isa SJ
        T = sqrt.(A ./ mec.δ)
        ∂TDiv∂A = Diagonal(1 ./ (2 * sqrt.(A ./ mec.δ)))
        ∂ADiv∂u = Diagonal(A) * mec.V
    elseif mec.stype isa BR
        T = A ./ sqrt.(mec.δ)
        ∂TDiv∂A = Diagonal(1 ./ sqrt.(mec.δ))
        ∂ADiv∂u = Diagonal(A .^ 2) * mec.V
    else
        error("Unsupported entropy type")
    end
    @show 2
    ∂SDiv∂A = -(Diagonal(mec.δ)*mec.V*u)'
    # construct Λ
    H = Diagonal(mec.δ) * mec.kernel' * mec.kernel * Diagonal(mec.δ)
    Λ = H * Diagonal(T) * invΛ(α, Diagonal(T) * H * Diagonal(T))^2

    ∂uαDiv∂α = func∂uαDiv∂α(mec)(u, α)
    ∂uαDiv∂G = func∂uαDiv∂G(mec)(u, α)
    tmp = (diag(Λ)' * ∂TDiv∂A + ∂SDiv∂A) * ∂ADiv∂u
    ∂αoptDiv∂G = -α/(S + α * tmp * ∂uαDiv∂α) * tmp * ∂uαDiv∂G
    ∂uoptDiv∂G = ∂uαDiv∂G + ∂uαDiv∂α * ∂αoptDiv∂G
    ∂AoptDiv∂G = ∂ADiv∂u * ∂uoptDiv∂G
    N = length(mec.Gᵥ) ÷ 2
    return A, ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

=#
