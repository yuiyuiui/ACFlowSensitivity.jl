mutable struct Solver
    method::String
    Atype::String
    Ward::String
end

struct CtxData{T<:Real}
    β::T
    wn::Vector{T}
    iwn::Vector{Complex{T}}
    mesh::Vector{T}
    mesh_weights::Vector{T}
    function CtxData(;
        β = 10.0::T,
        output_bound = 8.0::T,
        output_number = 801::Int64,
        mesh_type::Mesh = UniformMesh(),
    ) where {T}
        wn = (collect(0:(N-1)) .+ T(0.5)) * 2π / β
        iwn = (collect(0:(N-1)) .+ T(0.5)) * 2π / β * im
        mesh, mesh_weights = make_mesh(output_bound, output_number, mesh_type)
        return new(β, wn, iwn, mesh, mesh_weights)
    end
end

struct BarRat end

struct maxent
    ite::Int64
    method::String
end
