abstract type Mesh end

struct UniformMesh <: Mesh end

struct TangentMesh{T<:Real} <: Mesh
    p::T
    function TangentMesh(; p::T=2.1) where {T<:Real}
        return new{T}(p)
    end
end

function make_mesh(mb::T, ml::Int, mesh_type::UniformMesh) where {T<:Real}
    mesh=collect(range(-mb, mb, ml))
    mesh_weights = (mesh[2:end] + mesh[1:(end - 1)]) * 1//2
    pushfirst!(mesh_weights, mesh[1])
    push!(mesh_weights, mesh[end])
    mesh_weights = diff(mesh_weights)
    return mesh, mesh_weights
end

function make_mesh(mb::T, ml::Int, mesh_type::TangentMesh) where {T<:Real}
    p=T(mesh_type.p)
    mesh=tan.(collect(range(-T(π)/p, T(π)/p, ml)))
    mesh = mesh/mesh[end]*mb
    mesh_weights = (mesh[2:end] + mesh[1:(end - 1)]) * 1//2
    pushfirst!(mesh_weights, mesh[1])
    push!(mesh_weights, mesh[end])
    mesh_weights = diff(mesh_weights)
    return mesh, mesh_weights
end
