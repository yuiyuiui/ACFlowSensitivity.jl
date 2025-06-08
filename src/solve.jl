struct CtxData{T<:Real}
    β::T
    N::Int
    wn::Vector{T}
    iwn::Vector{Complex{T}}
    mesh::Vector{T}
    mesh_weights::Vector{T}
    η::T
    function CtxData(β::T,
                     N::Int;
                     mesh_bound=ACFSDefults.mesh_bound[]::Real,
                     mesh_length=ACFSDefults.mesh_length[]::Int,
                     mesh_type::Mesh=ACFSDefults.mesh_type[]::Mesh,
                     η::T=T(1e-2),) where {T<:Real}
        wn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β
        iwn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β * im
        mesh, mesh_weights = make_mesh(T(mesh_bound), mesh_length, mesh_type)
        return new{T}(β, N, wn, iwn, mesh, mesh_weights, η)
    end
end

abstract type SpectrumType end

struct Cont <: SpectrumType end
struct Delta <: SpectrumType end
struct Mixed <: SpectrumType end

# abanda add singular values of the lowner matrix less than `minsgl` for numerical stability
# this method is under developing
struct BarRat
    spt::SpectrumType
    minsgl::Real
    aaa_tol::Real
    max_degree::Int
    lookaheaad::Int
    function BarRat(spt::SpectrumType;
                    minsgl::Real=0,
                    aaa_tol::Real=ACFSDefults.tol[],
                    max_degree::Int=150,
                    lookaheaad::Int=10,)
        return new(spt, minsgl, aaa_tol, max_degree, lookaheaad)
    end
end

abstract type MaxEnt end

struct MaxEntChi2kink <: MaxEnt
    maxiter::Int
    L::Int
    α₁::Real
    σ::Real
    model_type::String
    function MaxEntChi2kink(;
                            maxiter::Int=1,
                            L::Int=16,
                            α₁::Real=1e12,
                            σ::Real=1e-4,
                            model_type::String="Gaussian",)
        return new(maxiter, L, α₁, σ, model_type)
    end
end

module ACFSDefults
using ..ACFlowSensitivity
const tol = Ref(1e-12)
const mesh_bound = Ref(8)
const mesh_length = Ref(801)
const mesh_type = Ref(UniformMesh())
end
