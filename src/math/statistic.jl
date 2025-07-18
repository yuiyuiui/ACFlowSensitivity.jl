function mean(x::Vector{T}) where {T}
    return sum(x) / length(x)
end

function mean(x::Vector{T}, w::Vector{T}) where {T}
    return sum(x .* w) / sum(w)
end

function median(v::Vector{T}) where {T}
    w = sort(v)
    n = length(w)
    if n % 2 == 0
        return (w[n÷2] + w[n÷2+1]) / 2
    else
        return w[n÷2+1]
    end
end
