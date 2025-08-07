# ================================
# SpectrumType
# ================================

abstract type SpectrumType end

struct Cont <: SpectrumType end
struct Delta <: SpectrumType end
struct Mixed <: SpectrumType end

# ================================
# Context Data
# ================================

struct CtxData{T<:Real}
    spt::SpectrumType
    β::T
    N::Int
    wn::Vector{T}
    iwn::Vector{Complex{T}}
    mesh::Mesh{T}
    η::T
    σ::T
    fp_ww::Real # find peaks window width
    fp_mp::Real # find peaks minimum peak height
    function CtxData(spt::SpectrumType,
                     β::T,
                     N::Int;
                     mesh_bound=ACFSDefults.mesh_bound[]::Real,
                     mesh_length=ACFSDefults.mesh_length[]::Int,
                     mesh_type::MeshMethod=ACFSDefults.mesh_type[]::MeshMethod,
                     η::T=T(1e-4),
                     σ::T=T(1e-4),
                     fp_ww::Real=T(0.01),
                     fp_mp::Real=T(0.1)) where {T<:Real}
        wn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β
        iwn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β * im
        mesh = make_mesh(T(mesh_bound), mesh_length, mesh_type)
        return new{T}(spt, β, N, wn, iwn, mesh, η, σ, fp_ww, fp_mp)
    end
end

# ================================
# Solver
# ================================

abstract type Solver end

# BarRat ==========================
# abanda add singular values of the lowner matrix less than `minsgl` for numerical stability
# this method is under developing
struct BarRat <: Solver
    minsgl::Real
    aaa_tol::Real
    max_degree::Int
    lookaheaad::Int
    denoisy::Bool
    prony_tol::Real
    pcut::Real
end

function BarRat(;
                minsgl::Real=0,
                aaa_tol::Real=ACFSDefults.tol[],
                max_degree::Int=150,
                lookaheaad::Int=10,
                denoisy::Bool=false,
                prony_tol::Real=-1,
                pcut::Real=1e-3)
    return BarRat(minsgl, aaa_tol, max_degree, lookaheaad, denoisy, prony_tol, pcut)
end

# NAC ==========================
mutable struct NAC <: Solver
    pick::Bool
    hardy::Bool
    hmax::Int
    alpha::Real
    eta::Real
end

function NAC(; pick=true,
             hardy=true,
             hmax=50,
             alpha=1e-4,
             eta=1e-2,)
    @info("For delta type spectrum, `pick = false` and `hardy = false` are recommended")
    return NAC(pick, hardy, hmax, alpha, eta)
end

# MaxEnt ==========================

struct MaxEnt <: Solver
    method::String
    stype::String
    nalph::Int
    alpha::Real
    ratio::Real
    model_type::String
    offdiag::Bool
    blur::Real
end
# what does `offdiag` in fact means?
function MaxEnt(;
                method::String="chi2kink",
                stype::String="sj",
                nalph::Int=12,
                alpha::Real=1e9,
                ratio::Real=10,
                model_type::String="Gaussian",
                offdiag::Bool=false,
                blur::Real=-1)
    return MaxEnt(method, stype, nalph, alpha, ratio, model_type, offdiag, blur)
end

# SSK ==========================

mutable struct SSK <: Solver
    nfine::Int
    npole::Int
    nwarm::Int
    nstep::Int
    retry::Int
    θ::Real
    ratio::Real
    method::String
end
function SSK(npole::Int;
             nfine::Int=100000,
             nwarm::Int=1000,
             nstep::Int=20000,
             retry::Int=10,
             θ::Real=1e6,
             ratio::Real=0.9,
             method::String="chi2min")
    return SSK(nfine, npole, nwarm, nstep, retry, θ, ratio, method)
end

# SAC ==========================

struct SAC <: Solver
    nfine::Int
    npole::Int
    nwarm::Int
    nstep::Int
    ndump::Int
    nalph::Int
    alpha::Real
    ratio::Real
end
function SAC(npole::Int;
             nfine::Int=10000,
             nwarm::Int=4000,
             nstep::Int=4000000,
             ndump::Int=40000,
             nalph::Int=20,
             alpha::Real=1.0,
             ratio::Real=1.2)
    return SAC(nfine, npole, nwarm, nstep, ndump, nalph, alpha, ratio)
end

# SOM ==========================
struct SOM <: Solver
    ntry::Int
    nstep::Int
    nbox::Int
    sbox::Real
    wbox::Real
end
function SOM(;
             ntry::Int=3000,
             nstep::Int=1000,
             nbox::Int=200,
             sbox::Real=0.0025,
             wbox::Real=0.02)
    return SOM(ntry, nstep, nbox, sbox, wbox)
end

# SPX ==========================
struct SPX <: Solver
    method::String
    nfine::Int
    npole::Int
    ntry::Int
    nstep::Int
    theta::Real
    eta::Real
end
function SPX(npole::Int;
             method::String="best",
             nfine::Int=100000,
             ntry::Int=1000,
             nstep::Int=10000,
             theta::Real=1e6,
             eta::Real=1e-4)
    return SPX(method, nfine, npole, ntry, nstep, theta, eta)
end

# solve differentiation
function pγdiff(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                alg::Solver; equalγ::Bool=true) where {T<:Real}
    N = ctx.N
    Aout, (rep, reγ) = solve(GFV, ctx, alg)
    n = length(rep)
    function f(p, γ, G)
        @assert length(p) == length(γ)
        res = 0
        for j in 1:N
            res += abs2(sum(γ ./ (ctx.iwn[j] .- p)) - G[j])
        end
        return res
    end
    if equalγ
        f₁(p, G) = Zygote.gradient(p₁ -> f(p₁, reγ, G), p)[1]
        f₁₁(p, G) = Zygote.jacobian(p₁ -> f₁(p₁, G), p)[1]
        f₁₂(p, G) = Zygote.jacobian(G₁ -> f₁(p, G₁), G)[1]
        return Aout, (rep, reγ),
               (-pinv(f₁₁(rep, GFV)) * f₁₂(rep, GFV), zeros(Complex{T}, n, N))
    else
        repγ = vcat(rep, reγ)
        g(pγ, G) = f(pγ[1:n], pγ[(n + 1):end], G)
        g₁(pγ, G) = Zygote.gradient(pγ₁ -> g(pγ₁, G), pγ)[1]
        g₁₁(pγ, G) = Zygote.jacobian(pγ₁ -> g₁(pγ₁, G), pγ)[1]
        g₁₂(pγ, G) = Zygote.jacobian(G₁ -> g₁(pγ, G₁), G)[1]
        ∂pγDiv∂G = -pinv(g₁₁(repγ, GFV)) * g₁₂(repγ, GFV)
        return Aout, (rep, reγ),
               (∂pγDiv∂G[1:n, :], ∂pγDiv∂G[(n + 1):end, :])
    end
end

# ================================
# Defaults
# ================================

module ACFSDefults
using ..ACFlowSensitivity
const tol = Ref(1e-12)
const mesh_bound = Ref(8)
const mesh_length = Ref(801)
const mesh_type = Ref(UniformMesh())
end
