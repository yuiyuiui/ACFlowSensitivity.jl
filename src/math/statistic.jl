function mean(x::Vector{T}) where {T}
    return sum(x) / length(x)
end

function mean(x::Vector{T}, w::Vector{T}) where {T}
    return sum(x .* w) / sum(w)
end
