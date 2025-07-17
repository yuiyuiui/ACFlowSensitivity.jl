function make_kernel(mesh::Vector{T}, wn::Vector{T}) where {T<:Real}
    nmesh = length(mesh)
    nwn = length(wn)
    kernel = zeros(T, nmesh, nwn)
    for i in 1:nmesh
        for j in 1:nwn
            kernel[j, i] = 1 / (im * wn[j] - mesh[i])
        end
    end
    return vcat(real(kernel), imag(kernel))
end
