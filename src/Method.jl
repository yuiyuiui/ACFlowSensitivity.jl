include("SelfDefMathMethod.jl")

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
            


#--------------------------

# aaa algorithm writen by myself


function generate_G_value_image(β::Float64,N::Int64, A)
    grid=(collect(0:N-1).+0.5)*2π/β
    step = 1e-3
    boundary = 20
    Int_stp_num = trunc(Int64, boundary / step)  
    n = length(grid)
    res = zeros(ComplexF64, n)
    
    for i = 1:n
        # 计算每个 \hat{G}(w_n)=\int_R A(x)/(iw_n-x)dx
        
        # 辛普森法的积分
        integral = 0.0
        for j = -Int_stp_num:Int_stp_num
            x = j * step
            coeff = 1.0
            
            # 使用辛普森法的权重
            if j == -Int_stp_num || j == Int_stp_num
                coeff = 1  # 端点的权重为 1
            elseif j % 2 == 0
                coeff = 2  # 偶数点的权重为 2
            else
                coeff = 4  # 奇数点的权重为 4
            end
            
            integral += coeff * A(x) / (im * grid[i] - x)
        end
        
        integral *= step / 3  # 最后乘以步长 / 3
        res[i] = integral
    end
    
    return res
end

function reconstruct_spectral_density(input_grid::Vector{ComplexF64},input_values::Vector{ComplexF64},
    output_bound::Float64,output_number::Int64)
    # construct output mesh
    mesh=make_mesh(output_bound,output_number)
    return mesh,my_aaa(input_grid,input_values) 
end


# make tangent mesh
function make_mesh(output_bound::Float64,output_number::Int64)
    p=2.1
    grid=tan.(collect(range(-π/p,π/p,output_number)))
    return grid/grid[end]*output_bound
end

# reconstructed green function on the complex palne
function construct_bary_func(w::Vector{ComplexF64},g::Vector{ComplexF64},v::Vector{ComplexF64})
    function y(x)
        return sum((w.*v)./(x.-g))/sum(w./(x.-g))
    end
    return y
end

function construct_bf_sd(w::Vector{ComplexF64},g::Vector{ComplexF64},v::Vector{ComplexF64})
    f0=construct_bary_func(w,g,v)
    function f1(x::Float64)
        return -imag(f0(x))/π
    end
    return f1
end

#my_aaa algorithm
function my_aaa(grid::Vector{ComplexF64},values::Vector{ComplexF64};
    max_degree=150,
    tol=1000*eps(Float64),
    lookaheaad=10
    )
    @assert length(grid)>0

    #preparation
    @assert length(grid)==length(values)
    m=length(grid)
    best_error=Inf
    best_n=0
    best_weight=0.0+im*0.0
    best_index=Int[]

    max_values=maximum(abs.(values))
    wait_index=Set(1:m)
    chosen_index=Int64[]

    C=Matrix{ComplexF64}(undef,m,m)
    L=Matrix{ComplexF64}(undef,m,m)
    R=zeros(m)

    #get the first node
    mean_value=sum(values)/m
    _, idx = findmax(abs(value - mean_value) for value in values)

    push!(chosen_index, idx)
    delete!(wait_index, idx)

    # begin iteration
    n=0
    while true
        n+=1
        active_grid=grid[chosen_index]
        active_values=values[chosen_index]
        @inbounds @fastmath for i in wait_index
            δ = grid[i] - active_grid[n]
            C[i, n] = iszero(δ) ? 1 / eps() : 1 / δ
            L[i, n] = (values[i] - active_values[n]) * C[i, n]
        end
        _, _, V=svd(L[collect(wait_index),1:n])
        w=V[:,end]

        wait_active_C=view(C,collect(wait_index),1:n)
        num=wait_active_C*(w.*active_values)
        den=wait_active_C*w
        R[collect(wait_index)]=abs.(values[collect(wait_index)]-num./den)
        error,next_index=findmax(R)

        # Do we choose this as best?
        if(error<best_error)
            best_error=error
            best_n=n
            best_weight=w
            best_index=copy(chosen_index)
        end

        # Do we end the iteration?
        if (best_error<tol*max_values)||(n>=max_degree)||
            ((n-best_n>=lookaheaad)&&(best_error<1e-2*max_values))
            break
        end

        push!(chosen_index,next_index)
        delete!(wait_index,next_index)
        R[next_index]=0
    end

    return best_weight,grid[best_index],values[best_index]

end


# ----------------
# function for examine effect of methods
function aaa_check(A;β::Float64=10.0,N::Int64=20,output_bound::Float64=5.0,output_number::Int64=801)
    
    #generate green function values on image axis
    G_value_image=generate_G_value_image(β,N,A)
    input_grid=im*(collect(0:N-1).+0.5)*2π/β;
    # calculate reconstruct output mesh and spectral density and green function
    mesh,(w,g,v)=reconstruct_spectral_density(input_grid,G_value_image,output_bound,output_number)
    return mesh,construct_bf_sd(w,g,v),construct_bary_func(w,g,v)
end



