function make_kernel(mesh::Vector{T}, grid::Vector{T};
                     grid_type::String="mats_freq",
                     β::Union{T,Missing}=missing) where {T<:Real}
    nmesh = length(mesh)
    ngrid = length(grid)
    if grid_type == "mats_freq"
        kernel = zeros(Complex{T}, ngrid, nmesh)
        for i in 1:ngrid
            for j in 1:nmesh
                kernel[i, j] = 1 / (im * grid[i] - mesh[j])
            end
        end
    elseif grid_type == "imag_time"
        kernel = zeros(T, ngrid, nmesh)
        for i in 1:ngrid
            for j in 1:nmesh
                kernel[i, j] = exp(-grid[i] * mesh[j]) / (1 + exp(-β * mesh[j]))
            end
        end
    end
    return kernel
end

"""
    make_blur(mesh::Vector{T}, A::Vector{T}, blur::T) where {T<:Real}

Try to blur the given spectrum `A`, which is defined in `mesh`. And `blur`
is the blur parameter.

### Arguments
* mesh -> Real frequency mesh.
* A    -> Spectral function.
* blur -> Blur parameter. It must be larger than 0.0.

### Returns
* A    -> It is updated in this function.
"""
function make_blur(mesh::Vector{T}, A::Vector{T}, blur::T) where {T<:Real}
    spl = CubicSplineInterpolation(A, mesh)

    bmesh, gaussian = make_gauss_peaks(blur)

    nsize = length(bmesh)
    nmesh = length(mesh)

    Mb = reshape(bmesh, (nsize, 1))
    Mx = reshape(gaussian, (nsize, 1))
    Mm = reshape(mesh, (1, nmesh))
    I = Mx .* spl.(Mm .+ Mb)

    for j in 1:nmesh
        A[j] = simpson(bmesh, view(I, :, j))
    end
end

"""
    make_gauss_peaks(blur::T) where {T<:Real}

Try to generate a gaussian peak along a linear mesh, whose energy range
is `[-5 * blur, +5 * blur]`. The number of mesh points is fixed to 201.

### Arguments
* blur -> This parameter is used to control the width of gaussian peak.

### Returns
* bmesh -> A linear mesh in [-5 * blur, 5 * blur].
* gaussian -> A gaussian peak at `bmesh`.
"""
function make_gauss_peaks(blur::T) where {T<:Real}
    @assert blur > T(0)
    nsize = 201
    bmesh = collect(LinRange(-T(5) * blur, T(5) * blur, nsize))
    norm = T(1) / (blur * sqrt(T(2) * T(π)))
    gaussian = norm * exp.(-T(1//2) * (bmesh / blur) .^ T(2))
    return bmesh, gaussian
end
