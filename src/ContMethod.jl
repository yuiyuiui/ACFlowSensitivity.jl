# generate values of G(iw_n)
function generate_G_values_cont(β::Float64,N::Int64, A)
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

# reconstruct spectral density
function reconstruct_spectral_density(input_grid::Vector{ComplexF64},input_values::Vector{ComplexF64})
    # construct output mesh
    w,g,v=my_aaa(input_grid,input_values)
    return (w,g,v), construct_bf_sd(w,g,v)
end


# reconstructed green function on the complex palne
function construct_bary_func(w::Vector{ComplexF64},g::Vector{ComplexF64},v::Vector{ComplexF64})
    function y(x)
        return sum((w.*v)./(x.-g))/sum(w./(x.-g))
    end
    return y
end

# re construct spectral density function on the whole complex palne
function construct_bf_sd(w::Vector{ComplexF64},g::Vector{ComplexF64},v::Vector{ComplexF64})
    f0=construct_bary_func(w,g,v)
    function f1(x::Float64)
        return -imag(f0(x))/π
    end
    return f1
end

# aaa algorithm writen by myself
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

function permu(v::Vector;direction=:left::Symbol)
    n=length(v)
    res=similar(v,n)
    if direction==:left
        for i=1:n-1
            res[i]=v[i+1]
        end
        res[n]=v[1]
    else
        for i=1:n-1
            res[i+1]=v[i]
        end
        res[i]=v[n]
    end
    return res
end

function permu_closest(v::Vector,target::Int64)
    N=length(v)
    res=similar(v,N)
    @assert target>=1&&target<=N
    idx=1
    res[1]=v[target]
    for k=1:N
        if target-k<=0||target+k>N
            break
        end
        res[idx+1]=v[target-k]
        res[idx+2]=v[target+k]
        idx+=2
    end
    if target<=(1+N)/2
        res[idx+1:N]=v[2*target:N]
    else
        res[idx+1:N]=v[2*target-N-1:-1:1]
    end
    return res
end

#---------
# AD
function AD_aaa(β,N)

    input_grid=im*(collect(0:N-1).+0.5)*2π/β;
    function y(G_values::ComplexF64)
        (w,g,v),_=reconstruct_spectral_density(input_grid,G_values)
        return construct_bary_func(w,g,v)(input_grid[1])
    end
    return y

end