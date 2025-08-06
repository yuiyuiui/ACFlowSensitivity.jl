tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

"""
    APC

Alias of Complex type (Arbitrary Precision Complex).

See also: [`API`](@ref), [`APF`](@ref).
"""
const APC = Complex{BigFloat}

"""
    reprod(mesh::Vector{T}, kernel::Matrix{T}, A::Vector{T}) where {T<:Real}

Try to reproduce the input data, which can be compared with the raw data
to see whether the analytic continuation is reasonable.

### Arguments
* mesh -> Real frequency mesh.
* kernel -> The kernel function.
* A -> The calculated spectral function, A(ω) or A(ω) / ω.

### Returns
* G -> Reconstructed correlators, G(τ) or G(iωₙ), Vector{F64}.

See also: [`AbstractMesh`](@ref).
"""
function reprod(mesh::Vector{T}, kernel::Matrix{T}, A::Vector{T}) where {T<:Real}
    ndim, nmesh = size(kernel)
    @assert nmesh == length(mesh) == length(A)

    @einsum KA[i, j] := kernel[i, j] * A[j]

    G = zeros(T, ndim)
    for i in 1:ndim
        G[i] = trapz(mesh, view(KA, i, :))
    end

    return G
end
