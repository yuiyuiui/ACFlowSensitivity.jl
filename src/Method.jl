include("math.jl")
include("aaa.jl")
include("ADaaa.jl")
include("maxent.jl")
include("ADmaxent.jl")
include("sac.jl")
include("ADsac.jl")

struct Solver
    method::String
    Atype::String
    Ward::String
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

