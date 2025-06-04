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
    alg::MaxEntChi2kink) where {T<:Real}
    L = alg.L
    α₁ = T(alg.α₁)
    σ = T(alg.σ)
    w = ctx.mesh_weights
    ss = SingularSpace(GFV, ctx.mesh, ctx.iwn)
    reA = make_model(alg.model_type, ctx)
    αvec = Vector{T}(undef, L)
    αvec[1] = α₁
    for i in 2:L
        αvec[i] = αvec[i-1] / 10
    end
    _, K, _, U, S, V = ss
    KDw = K * Diagonal(w)
    DS = Diagonal(S)
    DSUadDivσ² = DS*U'/σ^2 # to construct J = -V'∂Q/∂A
    S²VadDwDivσ² = DS^2 * V' * Diagonal(w)/σ^2 # to construct H = -V'∂²Q/∂A∂u = ∂J/∂u
    return PreComput(ss, reA, αvec, σ, DS, KDw, DSUadDivσ², S²VadDwDivσ²)
end
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T},
               alg::MaxEntChi2kink) where {T<:Real}
    alg.maxiter > 1 && error("maxiter>1 is not stable for cont spectrum solve")
    maxiter = alg.maxiter
    pc = PreComput(GFV, ctx, alg)
    reA = pc.model
    for i in 1:maxiter
        pc.model .= reA
        reA = chi2kink(pc)
    end
    return ctx.mesh, reA
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

function G2χ²vec(pc::PreComput{T}) where {T<:Real}
    G, _, n, _, _, V = pc.ss
    αvec = pc.αvec
    χ²vec = Vector{T}(undef, length(αvec))
    _χ² = MaxEnt_χ²(G, pc.KDw, pc.model, V, pc.σ)
    # Now solve the minimal with Newton method
    u_guess = zeros(T, n)
    u_opt_vec = Vector{Vector{T}}(undef, length(αvec))
    for i in 1:length(αvec)
        #@show i
        α = αvec[i]
        u_opt, call, _ = newton(MaxEnt_J(α, pc.DSUadDivσ², pc.KDw, pc.model, V, G),
                                MaxEnt_H(α, pc.S²VadDwDivσ², pc.model, V), u_guess)
        u_guess = copy(u_opt)
        u_opt_vec[i] = copy(u_opt)
        χ²vec[i] = _χ²(u_opt)
        #@show log10(α), log10(χ²_vec[i]), norm(J(u_opt, α)), call
    end
    idx = findall(isfinite, χ²vec)
    return u_opt_vec, χ²vec, idx
end
function χ²vec2αopt(χ²vec::Vector{T}, αvec::Vector{T}) where {T<:Real}
    # Now performe curve fit
    guess_fit = [T(0), T(5), T(2), T(0)]
    function fitfun(x, p)
        return @. p[1] + p[2] / (T(1) + exp(-p[4] * (x - p[3])))
    end
    p = curve_fit(fitfun, log10.(αvec), log10.(χ²vec), guess_fit).param
    # choose the inflection point as the best α
    # Parameter to prevent overfitting when fitting the curve
    adjust = T(5//2)
    αopt = 10^(p[3]-adjust/p[4])
    return αopt
end
function chi2kink(pc::PreComput{T}) where {T<:Real}
    _A = MaxEnt_A(pc.model, pc.ss.V)
    u_opt_vec, χ²vec, idx = G2χ²vec(pc)
    αopt = χ²vec2αopt(χ²vec[idx], pc.αvec[idx])
    u_guess = copy(u_opt_vec[findmin(abs.(pc.αvec .- αopt))[2]])
    u_opt, = newton(MaxEnt_J(αopt, pc.DSUadDivσ², pc.KDw, pc.model, pc.ss.V, pc.ss.G),
                    MaxEnt_H(αopt, pc.S²VadDwDivσ², pc.model, pc.ss.V), u_guess)
    # recover the A
    return _A(u_opt)
end

#---------------------------------
# solve differentiation
function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                   alg::MaxEntChi2kink) where {T<:Real}
    alg.maxiter > 1 &&
        error("maxiter>1 is not stable for cont spectrum solve differentiation")
    pc = PreComput(GFV, ctx, alg)
    reA, ∂reADiv∂G = chi2kink_diff(pc)
    return ctx.mesh, reA, ∂reADiv∂G, ∇L2loss(∂reADiv∂G, ctx.mesh_weights)[2]
end

function _∂χ²vecDiv∂G(pc::PreComput{T}) where {T<:Real}
    G, K, n, U, _, V = pc.ss
    αvec = pc.αvec
    σ = pc.σ
    model = pc.model
    N = size(K, 1) ÷ 2
    L = length(αvec)
    χ²vec = Vector{T}(undef, L) #χ²vec is χ²opt_vec
    _A = MaxEnt_A(model, V)
    _χ² = MaxEnt_χ²(G, pc.KDw, model, V, σ)
    # ∂A/∂u 
    _∂ADiv∂u(u::Vector{T}) = Diagonal(_A(u)) * V

    # ∂χ²/∂A, get a row matrix
    _∂χ²Div∂u(u::Vector{T}) = Matrix(2/σ^2 * (_A(u)'*pc.KDw' - G') * pc.KDw) * _∂ADiv∂u(u)

    # ∂J/∂G 
    ∂JDiv∂G = -1 / (σ^2) * pc.DS * U'

    # ∂χ²/∂G, get a row matrix
    _∂χ²Div∂G(u::Vector{T}) = Matrix(2 / (σ^2) * (G' - _A(u)' * K'))

    ∂χ²vecDiv∂G = Matrix{T}(undef, L, 2 * N)

    # 接下来用Newton method求最值点
    u_guess = zeros(T, n)
    u_opt_vec = Vector{Vector{T}}(undef, L)
    for i in 1:L
        α = αvec[i]
        _Hᵢ = MaxEnt_H(α, pc.S²VadDwDivσ², model, V)
        uᵢ, = newton(MaxEnt_J(α, pc.DSUadDivσ², pc.KDw, model, V, G), _Hᵢ, u_guess)
        u_guess = copy(uᵢ)
        u_opt_vec[i] = copy(uᵢ)
        χ²vec[i] = _χ²(uᵢ)
        if i == L && !all(isfinite, _A(uᵢ))
            χ²vec[i] = NaN
            ∂χ²vecDiv∂G = ∂χ²vecDiv∂G[1:(L - 1), :]
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
        return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
    end
    guess_fit = [T(0), T(5), T(2), T(0)]
    p = curve_fit(fitfun, log10.(αvec), log10.(χ²vec), guess_fit).param
    adjust = T(5//2)
    αopt = 10^(p[3]-adjust/p[4])
    ∂αoptDiv∂p = gradient((par->10^(par[3]-adjust/par[4])), p)[1]
    arg = (p, log10.(αvec), log10.(χ²vec))
    ∂pDiv∂χ²vec = -pinv(∂²lossϕDiv∂p²(arg...)) * ∂²lossϕDiv∂p∂y(arg...) *
                  Diagonal(1 ./ (χ²vec * T(log(10))))
    return Matrix(∂αoptDiv∂p') * ∂pDiv∂χ²vec, αopt
end
function chi2kink_diff(pc::PreComput{T}) where {T<:Real}
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
