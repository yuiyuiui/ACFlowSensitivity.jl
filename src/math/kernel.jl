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
