function find_peaks(v, minipeak)
    idx = findall(x -> x > minipeak, v)
    diff_right = vcat(v[1:(end - 1)]-v[2:end], v[end])
    diff_left = vcat(v[1], v[2:end]-v[1:(end - 1)])
    res = []
    for j in idx
        diff_right[j] >= 0 && diff_left[j] >= 0 && push!(res, j)
    end
    return res
end

function find_peaks(mesh, v, minipeak; wind=0.01)
    @assert length(mesh) == length(v)
    n = length(mesh)
    idx = findall(x -> x > minipeak, v)
    diff_right = vcat(v[1:(end - 1)]-v[2:end], v[end])
    diff_left = vcat(v[1], v[2:end]-v[1:(end - 1)])
    tmp = []
    res = []
    for j in idx
        diff_right[j] >= 0 && diff_left[j] >= 0 && push!(tmp, j)
    end
    for j in tmp
        flag = true
        k = 1
        while j+k <= n && abs(mesh[j+k] - mesh[j]) < wind
            if v[j+k] >= v[j]
                flag = false
                break
            end
            k += 1
        end
        k = 1
        while j-k >= 1 && abs(mesh[j] - mesh[j-k]) < wind
            if v[j-k] >= v[j]
                flag = false
                break
            end
            k += 1
        end
        flag && push!(res, j)
    end
    return res
end

#=
function poles2realγ(p::Vector, GFV::Vector{T}, iwn::Vector{T}) where {T<:Complex}
    ker = [1/(iwn[i] - p[j]) for i in 1:length(iwn), j in eachindex(p)]
    K = [real(ker); imag(ker)]
    G = vcat(real(GFV), imag(GFV))
    KtK = K'*K
    KtG = K'*G
    γ₀ = ones(real(T), length(p)) ./ length(p)
    γopt, _, _ = newton(x->KtK*x-KtG, x->KtK, γ₀)
    return γopt
end
=#

function pG2γ(x, y, iwn) # x is p, y is G
    ker = [1/(iwn[i] - x[j]) for i in 1:length(iwn), j in eachindex(x)]
    K = real(ker)'*real(ker) + imag(ker)'*imag(ker)
    b = real(ker)'*real(y) + imag(ker)'*imag(y)
    return pinv(K)*b
end
