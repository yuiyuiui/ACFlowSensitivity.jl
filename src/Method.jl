include("ContMethod.jl")
include("DeltaMethod.jl")


# function for examine effect of methods
function aaa_check(A;β::Float64=10.0,N::Int64=20,output_bound::Float64=5.0,output_number::Int64=801)
    
    #generate green function values on image axis
    G_values_image=generate_G_values_cont(β,N,A)
    input_grid=im*(collect(0:N-1).+0.5)*2π/β;

    # calculate reconstruct output mesh and spectral density and green function
    mesh=make_mesh(output_bound,output_number)
    (w,g,v),reA=reconstruct_spectral_density(input_grid,G_values_image)
    return mesh,reA,construct_bary_func(w,g,v)
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

# for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0],[ε],[1/(sqrt(2π)*ε)])
end

# make tangent mesh
function make_mesh(output_bound::Float64,output_number::Int64)
    p=2.1
    grid=tan.(collect(range(-π/p,π/p,output_number)))
    return grid/grid[end]*output_bound
end



#---------------------
# method for study limination of Discrete freen function


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

