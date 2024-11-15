struct BarycentricFunction<:Function
    nodes::Vector{C64}
    values::Vector{C64}
    weights::Vector{C64}
end

function (r::BarycentricFunction)(z::Number)
    w_times_f = r.values .* r.weights
    if isinf(z)
        return sum(w_times_f) / sum(r.weights)
    end
    #
    # Try to determine whether z is a valid node
    k = findfirst(z .== r.nodes)
    #
    if isnothing(k) # Not at a node
        C =  1 ./ (z .- r.nodes)
        return sum(C .* w_times_f) / sum(C .* r.weights)
    else            # Interpolation at node
        return r.values[k]
    end
end


# Perform once aaa algorithm and get constants needed for ADaaa
struct ADaaaBase
    wn::Vector{Float64}
    iwn::Vector{ComplexF64}
    Giwn::Vector{ComplexF64}
    Index0::Vector{Vector{Int64}}
    brcF::BarycentricFunction
end

function ADaaaBase(wn::Vector{Float64},Giwn::Vector{ComplexF64})
    @assert length(wn)==length(Giwn)
    w,g,v,bi=aaa(im*wn,Giwn;isAD=true)
    brcF=BarycentricFunction(w,g,v)
    Index0=[setdiff(1:length(wn),bi),bi]
    return ADaaaBase(wn,im*wn,Giwn,Index0,brcF)
end

function ADaaa(wn::Vector{Float64},Giwn::Vector{ComplexF64})
    @assert length(wn)==length(Giwn)
    ada=ADaaaBase(wn,Giwn)

    return ada
end



