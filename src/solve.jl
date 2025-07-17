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
    mesh::Vector{T}
    mesh_weights::Vector{T}
    η::T
    σ::T
    fp_ww::Real # find peaks window width
    fp_mp::Real # find peaks minimum peak height
    function CtxData(spt::SpectrumType,
                     β::T,
                     N::Int;
                     mesh_bound=ACFSDefults.mesh_bound[]::Real,
                     mesh_length=ACFSDefults.mesh_length[]::Int,
                     mesh_type::Mesh=ACFSDefults.mesh_type[]::Mesh,
                     η::T=T(1e-4),
                     σ::T=T(1e-4),
                     fp_ww::Real=T(0.01),
                     fp_mp::Real=T(0.1)) where {T<:Real}
        wn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β
        iwn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β * im
        mesh, mesh_weights = make_mesh(T(mesh_bound), mesh_length, mesh_type)
        return new{T}(spt, β, N, wn, iwn, mesh, mesh_weights, η, σ, fp_ww, fp_mp)
    end
end

# ================================
# Solver
# ================================

abstract type Solver end
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
    function BarRat(;
                    minsgl::Real=0,
                    aaa_tol::Real=ACFSDefults.tol[],
                    max_degree::Int=150,
                    lookaheaad::Int=10,
                    denoisy::Bool=false,
                    prony_tol::Real=-1,
                    pcut::Real=1e-3)
        return new(minsgl, aaa_tol, max_degree, lookaheaad, denoisy, prony_tol, pcut)
    end
end

# MaxEnt ==========================

abstract type MaxEnt <: Solver end

struct MaxEntChi2kink <: MaxEnt
    maxiter::Int
    L::Int
    α₁::Real
    model_type::String
    function MaxEntChi2kink(;
                            maxiter::Int=1,
                            L::Int=16,
                            α₁::Real=1e12,
                            model_type::String="Gaussian",)
        return new(maxiter, L, α₁, model_type)
    end
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

struct SAC
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
