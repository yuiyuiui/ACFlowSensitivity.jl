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

struct SingularSpace{T<:Real}
    G::Vector{T}
    K::Matrix{T}
    n::Int
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
end
function SingularSpace(GFV::Vector{Complex{T}}, mesh::Vector{T},
                       iwn::Vector{Complex{T}}) where {T<:Real}
    kernel = Matrix{Complex{T}}(undef, length(GFV), length(mesh))
    for i in 1:length(GFV)
        for j in 1:length(mesh)
            kernel[i, j] = 1 / (iwn[i] - mesh[j])
        end
    end
    G = vcat(real(GFV), imag(GFV))
    K = [real(kernel); imag(kernel)]
    U, S, V = svd(K)
    n = count(x -> (x >= strict_tol(T)), S)
    V = V[:, 1:n]
    U = U[:, 1:n]
    S = S[1:n]
    return SingularSpace(G, K, n, U, S, V)
end
function Base.iterate(ss::SingularSpace)
    return (ss.G, 1)
end
function Base.iterate(ss::SingularSpace, state)
    state == 1 && return (ss.K, 2)
    state == 2 && return (ss.n, 3)
    state == 3 && return (ss.U, 4)
    state == 4 && return (ss.S, 5)
    state == 5 && return (ss.V, 6)
    return nothing
end
