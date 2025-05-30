abstract type Mesh{T<:Real} end

struct UniformMesh{T<:Real} <: Mesh{T} end

struct TangentMesh{T<:Real} <: Mesh{T}
    p::T
    function TangentMesh(p::T) where {T<:Real}
        return new{T}(p)
    end
end

function make_mesh(mb::T, ml::Int, mesh_type::UniformMesh{T}) where {T<:Real}
    mesh=collect(range(-mb, mb, ml))
    mesh_weights = (mesh[2:end] + mesh[1:(end-1)]) * 1//2
    pushfirst!(mesh_weights, mesh[1])
    push!(mesh_weights, mesh[end])
    mesh_weights = diff(mesh_weights)
    return mesh, mesh_weights
end

function make_mesh(mb::T, ml::Int, mesh_type::TangentMesh{T}) where {T<:Real}
    p=mesh_type.p
    mesh=tan.(collect(range(-T(π)/p, T(π)/p, ml)))
    mesh = mesh/mesh[end]*mb
    mesh_weights = (mesh[2:end] + mesh[1:(end-1)]) * 1//2
    pushfirst!(mesh_weights, mesh[1])
    push!(mesh_weights, mesh[end])
    mesh_weights = diff(mesh_weights)
    return mesh, mesh_weights
end
