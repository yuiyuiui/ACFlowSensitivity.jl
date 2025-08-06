# ================================
# diff chi2kink
# ================================

struct PreComput{T<:Real}
    ss::SingularSpace{T}
    model::Vector{T}
    αvec::Vector{T}
    σ::T
    DS::Diagonal{T,Vector{T}}
    KDw::Matrix{T}
    DSUadDivσ²::Matrix{T}
    S²VadDwDivσ²::Matrix{T}
end
function PreComput(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                   alg::MaxEnt) where {T<:Real}
    nalph = alg.nalph
    α₁ = T(alg.alpha)
    ratio = T(alg.ratio)
    σ = T(ctx.σ)
    w = ctx.mesh.weight
    ss = SingularSpace(GFV, ctx.iwn, ctx.mesh.mesh)
    reA = make_model(alg.model_type, ctx)
    αvec = Vector{T}(undef, nalph)
    αvec[1] = α₁
    for i in 2:nalph
        αvec[i] = αvec[i-1] / ratio
    end
    _, K, _, U, S, V = ss
    KDw = K * Diagonal(w)
    DS = Diagonal(S)
    DSUadDivσ² = DS*U'/σ^2 # to construct J = -V'∂Q/∂A
    S²VadDwDivσ² = DS^2 * V' * Diagonal(w)/σ^2 # to construct H = -V'∂²Q/∂A∂u = ∂J/∂u
    return PreComput(ss, reA, αvec, σ, DS, KDw, DSUadDivσ², S²VadDwDivσ²)
end

struct MaxEnt_A{T<:Real} <: Function
    model::Vector{T}
    V::Matrix{T}
end
(f::MaxEnt_A{T})(u::Vector{T}) where {T<:Real} = f.model .* exp.(f.V * u)
struct MaxEnt_χ²{T<:Real} <: Function
    G::Vector{T}
    KDw::Matrix{T}
    model::Vector{T}
    V::Matrix{T}
    σ::T
end
function (f::MaxEnt_χ²{T})(u::Vector{T}) where {T<:Real}
    res = (f.G - f.KDw * (f.model .* exp.(f.V * u))) / (f.σ)
    return res' * res
end
struct MaxEnt_J{T<:Real} <: Function
    α::T
    DSUadDivσ²::Matrix{T}
    KDw::Matrix{T}
    model::Vector{T}
    V::Matrix{T}
    G::Vector{T}
end
function (f::MaxEnt_J{T})(u::Vector{T}) where {T<:Real}
    return f.α * u + f.DSUadDivσ² * (f.KDw * (f.model .* exp.(f.V * u)) - f.G)
end
struct MaxEnt_H{T<:Real} <: Function
    α::T
    S²VadDwDivσ²::Matrix{T}
    model::Vector{T}
    V::Matrix{T}
end
function (f::MaxEnt_H{T})(u::Vector{T}) where {T<:Real}
    return f.α * I(size(f.V, 2)) + f.S²VadDwDivσ² * Diagonal(f.model .* exp.(f.V * u)) * f.V
end

function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                   alg::MaxEnt) where {T<:Real}
    alg.maxiter > 1 &&
        error("maxiter>1 is not stable for cont spectrum solve differentiation")
    if ctx.spt isa Cont
        alg.method == "chi2kink" && (reA, ∂reADiv∂G=chi2kink_diff(GFV, ctx, alg))
        return reA, ∂reADiv∂G
    elseif ctx.spt isa Delta
        return pγdiff(GFV, ctx, alg; equalγ=false)
    else
        error("Unsupported spectral function type")
    end
end

function chi2kink_diff(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                       alg::MaxEnt) where {T<:Real}
    pc = PreComput(GFV, ctx, alg)
    ss = pc.ss
    model = pc.model
    G, _, _, U, _, V = pc.ss
    ∂χ²vecDiv∂G, u_opt_vec, χ²vec, idx = _∂χ²vecDiv∂G(pc)
    ∂αoptDiv∂χ²vec, αopt = _∂αoptDiv∂χ²vec(χ²vec, pc, idx)
    ∂αoptDiv∂G = ∂αoptDiv∂χ²vec * ∂χ²vecDiv∂G

    u_guess = copy(u_opt_vec[findmin(abs.(log10.(pc.αvec) .- log10(αopt)))[2]])
    _Hopt = MaxEnt_H(αopt, pc.S²VadDwDivσ², model, V)
    u_opt, = newton(MaxEnt_J(αopt, pc.DSUadDivσ², pc.KDw, model, V, G),
                    _Hopt, u_guess)

    N = size(ss.K, 1) ÷ 2
    _A = MaxEnt_A(model, V)
    _∂ADiv∂u(u::Vector{T}) = Diagonal(_A(u)) * V
    ∂JDiv∂G = -1 / (pc.σ^2) * pc.DS * U'
    _∂JDiv∂α(u::Vector{T}) = u
    ∂u_optDiv∂G = -pinv(_Hopt(u_opt)) * (_∂JDiv∂α(u_opt) * ∂αoptDiv∂G + ∂JDiv∂G)
    ∂AoptDiv∂G = _∂ADiv∂u(u_opt) * ∂u_optDiv∂G
    return _A(u_opt), ∂AoptDiv∂G[:, 1:N] + im * ∂AoptDiv∂G[:, (N + 1):(2 * N)]
end

function _∂χ²vecDiv∂G(pc::PreComput{T}) where {T<:Real}
    G, K, n, U, _, V = pc.ss
    αvec = pc.αvec
    σ = pc.σ
    model = pc.model
    N = size(K, 1) ÷ 2
    nalph = length(αvec)
    χ²vec = Vector{T}(undef, nalph) #χ²vec is χ²opt_vec
    _A = MaxEnt_A(model, V)
    _χ² = MaxEnt_χ²(G, pc.KDw, model, V, σ)
    # ∂A/∂u 
    _∂ADiv∂u(u::Vector{T}) = Diagonal(_A(u)) * V

    # ∂χ²/∂A, get a row matrix
    _∂χ²Div∂u(u::Vector{T}) = Matrix(2/σ^2 * (_A(u)'*pc.KDw' - G') * pc.KDw) * _∂ADiv∂u(u)

    # ∂J/∂G 
    ∂JDiv∂G = -1 / (σ^2) * pc.DS * U'

    # ∂χ²/∂G, get a row matrix
    _∂χ²Div∂G(u::Vector{T}) = Matrix(2 / (σ^2) * (G' - _A(u)' * pc.KDw'))

    ∂χ²vecDiv∂G = Matrix{T}(undef, nalph, 2 * N)

    # then use Newton method to find the minimum point
    u_guess = zeros(T, n)
    u_opt_vec = Vector{Vector{T}}(undef, nalph)
    for i in 1:nalph
        α = αvec[i]
        _Hᵢ = MaxEnt_H(α, pc.S²VadDwDivσ², model, V)
        uᵢ, = newton(MaxEnt_J(α, pc.DSUadDivσ², pc.KDw, model, V, G), _Hᵢ, u_guess)
        u_guess = copy(uᵢ)
        u_opt_vec[i] = copy(uᵢ)
        χ²vec[i] = _χ²(uᵢ)
        if i == nalph && !all(isfinite, _A(uᵢ))
            χ²vec[i] = NaN
            ∂χ²vecDiv∂G = ∂χ²vecDiv∂G[1:(nalph - 1), :]
            break
        end
        ∂χ²vecDiv∂G[i, :] = -_∂χ²Div∂u(uᵢ) * pinv(_Hᵢ(uᵢ)) * ∂JDiv∂G + _∂χ²Div∂G(uᵢ) # ∂χ²opt/∂G
    end
    idx = findall(isfinite, χ²vec)
    return ∂χ²vecDiv∂G, u_opt_vec, χ²vec, idx
end
function _∂αoptDiv∂χ²vec(χ²vec::Vector{T}, pc::PreComput{T},
                         idx::Vector{Int}) where {T<:Real}
    χ²vec = χ²vec[idx]
    αvec = pc.αvec[idx]
    function fitfun(x, p)
        return @. p[1] + p[2] / (T(1) + exp(-p[4] * (x - p[3])))
    end
    guess_fit = [T(0), T(5), T(2), T(0)]
    p = curve_fit(fitfun, log10.(αvec), log10.(χ²vec), guess_fit).param
    adjust = T(5//2)
    αopt = 10^(p[3]-adjust/p[4])
    ∂αoptDiv∂p = Zygote.gradient((par->10^(par[3]-adjust/par[4])), p)[1]
    arg = (p, log10.(αvec), log10.(χ²vec))
    ∂pDiv∂χ²vec = -pinv(∂²lossϕDiv∂p²(arg...)) * ∂²lossϕDiv∂p∂y(arg...) *
                  Diagonal(1 ./ (χ²vec * T(log(10))))
    return Matrix(∂αoptDiv∂p') * ∂pDiv∂χ²vec, αopt
end
