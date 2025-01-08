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
    for i=1:n
        res[i]+=noise*rand()*res[i]*exp(2π*im*rand())
    end
    return res
end


# ------------------------------------------------------------------------------------

include("math.jl")
include("aaa.jl")
include("ADaaa.jl")
include("maxent.jl")
include("ADmaxent.jl")
include("sac.jl")
include("ADsac.jl")



# ------------------------------------------------------------------------------------


# methods for proving decrete tends to continous one
function allowed_mesh(grid::Vector{Float64},mesh::Vector{Float64},dis::Float64)
    mesh1=zeros(length(mesh))
    n=0;i=1;j=1
    while true
        if j==length(mesh)+1
            break
        end

        if abs(grid[i]-mesh[j])<dis
            j+=1
            continue
        elseif mesh[j]<grid[i]
            n+=1
            mesh1[n]=mesh[j]
            j+=1
        elseif i<length(grid)
            i+=1
        else 
            n+=1
            mesh1[n]=mesh[j]
            j+=1
        end
    end
    res=mesh1[1:n]
    return res
end

function discrete_GF(grid::Vector{Float64},A_value::Vector{Float64},mesh; ϵ=eps(Float64))
    δ=grid[2]-grid[1]
    res=zeros(length(mesh))
    for i=1:length(mesh)
        for j=1:length(grid)
            x = (mesh[i]-grid[j])
            
            res[i]+=A_value[j]*δ*x/(x^2 + ϵ^2)
        end
    end
    return res
end

function benchmark_GF(A, oup_b; output_step=1e-3,calculate_step=1e-2)
    #输出的网格
    grid=collect(-oup_b:output_step:oup_b)

    #计算积分的网格
    grid1=collect(calculate_step:calculate_step:2*oup_b+2)

    res=zeros(length(grid))
    idx=0
    for w in grid
        idx+=1
        for x in grid1
            res[idx]+=(A(w-x)-A(w+x))/x*calculate_step
        end
    end
    return (grid,res)
end

