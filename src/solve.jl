struct CtxData{T<:Real}
    β::T
    N::Int
    wn::Vector{T}
    iwn::Vector{Complex{T}}
    mesh::Vector{T}
    mesh_weights::Vector{T}
    η::T
    function CtxData(
        β::T,
        N::Int;
        mesh_bound = ACFSDefults.mesh_bound[]::Real,
        mesh_length = ACFSDefults.mesh_length[]::Int,
        mesh_type::Mesh = ACFSDefults.mesh_type[]::Mesh,
        η::T = T(1e-2),
    ) where {T<:Real}
        wn = (collect(0:(N-1)) .+ T(0.5)) * T(2π) / β
        iwn = (collect(0:(N-1)) .+ T(0.5)) * T(2π) / β * im
        mesh, mesh_weights = make_mesh(T(mesh_bound), mesh_length, mesh_type)
        return new{T}(β, N, wn, iwn, mesh, mesh_weights, η)
    end
end

struct BarRat
    aaa_tol::Real
    max_degree::Int
    lookaheaad::Int
    function BarRat(;
        aaa_tol = ACFSDefults.tol[]::Real,
        max_degree = ACFSDefults.max_degree[]::Int,
        lookaheaad = ACFSDefults.lookaheaad[]::Int,
    )
        return new(aaa_tol, max_degree, lookaheaad)
    end
end

struct maxent
    ite::Int
    method::String
end

module ACFSDefults
using ..ACFlowSensitivity
const tol = Ref(1e-12)
const max_degree = Ref(150)
const lookaheaad = Ref(10)
const mesh_bound = Ref(8)
const mesh_length = Ref(801)
const mesh_type = Ref(UniformMesh())
end
