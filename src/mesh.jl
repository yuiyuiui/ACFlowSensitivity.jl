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
    mesh_weight = (mesh[2:end] + mesh[1:(end - 1)]) * 1//2
    pushfirst!(mesh_weight, mesh[1])
    push!(mesh_weight, mesh[end])
    mesh_weight = diff(mesh_weight)
    return mesh, mesh_weight
end

function make_mesh(mb::T, ml::Int, mesh_type::TangentMesh) where {T<:Real}
    p=T(mesh_type.p)
    mesh=tan.(collect(range(-T(π)/p, T(π)/p, ml)))
    mesh = mesh/mesh[end]*mb
    mesh_weight = (mesh[2:end] + mesh[1:(end - 1)]) * 1//2
    pushfirst!(mesh_weight, mesh[1])
    push!(mesh_weight, mesh[end])
    mesh_weight = diff(mesh_weight)
    return mesh, mesh_weight
end

struct SingularSpace{T<:Real}
    G::Vector{T}
    K::Matrix{T}
    n::Int
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
end
function SingularSpace(GFV::Vector{Complex{T}},
                       iwn::Vector{Complex{T}},
                       mesh::Vector{T}) where {T<:Real}
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

"""
    nearest(mesh::Vector{T}, r::Real) where {T<:Real}

Given a position `r` (0.0 ≤ r ≤ 1.0), and return the index of the nearest
point in the mesh `am`.

### Arguments
See above explanations.

### Returns
See above explanations.

### Examples
```julia
am = LinearMesh(1001, -10.0, 10.0)
pos = nearest(am, 0.2) # pos = 201
println(am[pos]) # -6.0
```

See also: [`AbstractMesh`](@ref).
"""
function nearest(mesh::Vector{T}, r::Real) where {T<:Real}
    # Check r and evaluate the corresponding value
    @assert 0 ≤ r ≤ 1
    val = mesh[1] + (mesh[end] - mesh[1]) * r

    # Try to locate val in the mesh by using the bisection algorithm
    left = 1
    right = length(mesh)
    @assert mesh[left] ≤ val ≤ mesh[right]

    while right - left ≥ 2
        mid = round(Int, (left + right) / 2)
        if val < mesh[mid]
            right = mid
        else
            left = mid
        end
    end

    # Well, now we have the left and right boundaries. We should return
    # the closer one.
    if mesh[right] - val > val - mesh[left]
        return left
    else
        return right
    end
end
