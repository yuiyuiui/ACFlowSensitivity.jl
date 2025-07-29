function slicerightmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    A[:, :, p] .= A[:, :, p] * B
    return A
end

function slicerightmul!_back(A::Array{T,3}, B::Matrix{T}, p::Int, dC::Array{T,3}) where {T}
    dA = deepcopy(dC)
    dA[:, :, p] .= dC[:, :, p] * B'
    dB = A[:, :, p]' * dC[:, :, p]
    return dA, dB
end

function sliceleftmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    A[:, :, p] .= B * A[:, :, p]
    return A
end

function sliceleftmul!_back(A::Array{T,3}, B::Matrix{T}, p::Int, dC::Array{T,3}) where {T}
    dA = deepcopy(dC)
    dB = dC[:, :, p] * A[:, :, p]'
    dA[:, :, p] .= B' * dC[:, :, p]
    return dA, dB
end

Zygote.@adjoint function slicerightmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    C = slicerightmul!(A, B, p)
    function pullback(dC)
        dA, dB = slicerightmul!_back(A, B, p, dC)
        return (dA, dB, nothing)
    end
    return C, pullback
end

Zygote.@adjoint function sliceleftmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    C = sliceleftmul!(A, B, p)
    function pullback(dC)
        dA, dB = sliceleftmul!_back(A, B, p, dC)
        return (dA, dB, nothing)
    end
    return C, pullback
end
