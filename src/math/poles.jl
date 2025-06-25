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
