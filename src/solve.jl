struct CtxData{T<:Real}
    β::T
    wn::Vector{T}
    iwn::Vector{Complex{T}}
    mesh::Vector{T}
    mesh_weights::Vector{T}
    function CtxData(β::T;
        mesh_bound = ACFSDefults.mesh_bound[],
        mesh_length = ACFSDefults.mesh_length[],
        mesh_type::Mesh = ACFSDefults.mesh_type[],
    ) where {T}
        wn = (collect(0:(N-1)) .+ T(0.5)) * T(2π) / β
        iwn = (collect(0:(N-1)) .+ T(0.5)) * T(2π) / β * im
        mesh, mesh_weights = make_mesh(mesh_bound, mesh_length, mesh_type)
        return new{T}(β, wn, iwn, mesh, mesh_weights)
    end
end

struct BarRat{T}
    aaa_tol::T
    max_degree::Int
    lookaheaad::Int
    function BarRat(;
        aaa_tol = ACFSDefults.tol[],
        max_degree = ACFSDefults.max_degree[],
        lookaheaad = ACFSDefults.lookaheaad[],
    )
        return new{T}(aaa_tol, max_degree, lookaheaad)
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
const mesh_type = Ref(UniformMesh{Float64}())
end
