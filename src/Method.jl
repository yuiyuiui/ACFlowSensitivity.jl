mutable struct Solver
    method::String
    Atype::String
    Ward::String
end

# construct combanition of gauss waves
function continous_spectral_density(μ::Vector{Float64},σ::Vector{Float64},peak::Vector{Float64})
    @assert length(μ)==length(σ)==length(peak)
    n=length(μ)
    function y(x::Float64)
        res=0
        for i=1:n
            res+=peak[i]*exp(-(x-μ[i])^2/(2*σ[i]^2))
        end
        return res
    end
    return y
end

# generate values of G(iw_n)
function generate_G_values_cont(β::Float64,N::Int64, A;int_low::Float64=-20.0,int_up::Float64=20.0,noise::Float64=0.0)
    grid=(collect(0:N-1).+0.5)*2π/β  
    n = length(grid)
    res = zeros(ComplexF64, n)
    for i = 1:n
        res[i] = quadgk(x -> A(x) / (im * grid[i] - x), int_low, int_up)[1]
    end
    NL = Normal(0.0,1.0)   # Normal list
    for i=1:n
        res[i]+=noise*rand(NL)*res[i]*exp(2π*im*rand())
    end
    return res
end

function generate_G_values_delta(β::Float64, N::Int64, poles::Vector{Float64}, γ_vec::Vector{Float64};noise::Float64=0.0)
    @assert length(poles) == length(γ_vec)
    wn=(collect(0:N-1).+0.5)*2π/β  
    res = zeros(ComplexF64, N)
    for i=1:N
        for j=1:length(poles)
            res[i] += γ_vec[j] / (im*wn[i]-poles[j])
        end
    end

    NL = Normal(0.0,1.0)

    res += noise * rand(NL,N) .* exp.(2π * im * rand(N))
    return res
end

# ------------------------------------------------------------------------------------

include("math.jl")
include("aaa.jl")
include("ADaaa.jl")
include("maxent.jl")
include("sac.jl")
include("ADsac.jl")

