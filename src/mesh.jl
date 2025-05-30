abstract type Mesh{T<:Real} end

struct UniformMesh{T<:Real} <: Mesh{T} end

struct TangentMesh{T<:Real} <: Mesh{T}
    p::T
    function TangentMesh(p::T) where {T<:Real}
        return new{T}(p)
    end
end

function make_mesh(opb::T, opn::Int64, mesh_type::UniformMesh{T}) where {T<:Real}
    mesh=collect(range(-opb, opb, opn))
    mesh_weights = (mesh[2:end] + mesh[1:(end-1)]) * 1//2
    pushfirst!(mesh_weights, mesh[1])
    push!(mesh_weights, mesh[end])
    mesh_weights = diff(mesh_weights)
    return mesh, mesh_weights
end

function make_mesh(opb::T, opn::Int64, mesh_type::TangentMesh{T}) where {T<:Real}
    p=mesh_type.p
    mesh=tan.(collect(range(-T(π)/p, T(π)/p, opn)))
    mesh = mesh/mesh[end]*opb
    mesh_weights = (mesh[2:end] + mesh[1:(end-1)]) * 1//2
    pushfirst!(mesh_weights, mesh[1])
    push!(mesh_weights, mesh[end])
    mesh_weights = diff(mesh_weights)
    return mesh, mesh_weights
end
