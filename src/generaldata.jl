# construct combanition of gauss waves
function continous_spectral_density(
    μ::Vector{T},
    σ::Vector{T},
    amplitude::Vector{T},
) where {T<:Real}
    @assert length(μ)==length(σ)==length(amplitude)
    n=length(μ)
    function y(x::T)
        res=T(0)
        for i = 1:n
            res+=amplitude[i]*exp(-(x-μ[i])^2/(2*σ[i]^2))
        end
        return res
    end
    return y
end

# generate values of G(iw_n)
function generate_GFV_cont(
    β::T,
    N::Int,
    A::Function;
    int_low::T = -T(20),
    int_up::T = T(20),
    noise::T = T(0),
) where {T<:Real}
    grid=(collect(0:(N-1)) .+ 1//2)*T(2π)/β
    n = length(grid)
    res = zeros(Complex{T}, n)
    for i = 1:n
        res[i] = integral(x -> A(x) / (im * grid[i] - x), int_low, int_up)
    end
    for i = 1:n
        res[i]+=noise*randn(T)*res[i]*exp(T(2π)*im*rand(T))
    end
    return res
end

function generate_GFV_delta(
    β::T,
    N::Int,
    poles::Vector{T},
    γ_vec::Vector{T};
    noise::T = T(0),
) where {T<:Real}
    @assert length(poles) == length(γ_vec)
    wn=(collect(0:(N-1)) .+ 1//2)*T(2π)/β
    res = zeros(Complex{T}, N)
    for i = 1:N
        for j = 1:length(poles)
            res[i] += γ_vec[j] / (im*wn[i]-poles[j])
        end
    end
    for i = 1:N
        res[i]+=noise*randn(T)*res[i]*exp(T(2π)*im*rand(T))
    end
    return res
end
