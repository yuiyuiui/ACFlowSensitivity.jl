# Aaa algorithm for continuous spectral density

# Main functions

# function for examine effect of methods
function aaa_check(A;β::Float64=10.0,N::Int64=20,output_bound::Float64=5.0,output_number::Int64=801,noise=0.0::Float64)
    
    #generate green function values on image axis
    G_values_image=generate_G_values_cont(β,N,A)
    for i in eachindex(G_values_image)
        G_values_image[i]+=G_values_image[i]*noise*rand()*exp(2π*im*rand())
    end
    input_grid=im*(collect(0:N-1).+0.5)*2π/β;

    # calculate reconstruct output mesh and spectral density and green function
    mesh=make_mesh(output_bound,output_number)
    (w,g,v),reA=reconstruct_spectral_density(input_grid,G_values_image)
    return mesh,reA,construct_bary_func(w,g,v)
end

# generate values of G(iw_n)
function generate_G_values_cont(β::Float64,N::Int64, A;int_low::Float64=-20.0,int_up::Float64=20.0)
    grid=(collect(0:N-1).+0.5)*2π/β  
    n = length(grid)
    res = zeros(ComplexF64, n)
    for i = 1:n
        res[i] = quadgk(x -> A(x) / (im * grid[i] - x), int_low, int_up)[1]
    end
    return res
end


# reconstruct spectral density
function reconstruct_spectral_density(input_grid::Vector{ComplexF64},input_values::Vector{ComplexF64})
    # construct output mesh
    w,g,v=my_aaa(input_grid,input_values)
    return (w,g,v), construct_bf_sd(w,g,v)
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

# aaa algorithm writen by myself
function my_aaa(grid::Vector{ComplexF64},values::Vector{ComplexF64};
    max_degree=150,
    tol=1000*eps(Float64),
    lookaheaad=10,
    isAD::Bool=false
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

        _, S, V=svd(L[collect(wait_index),1:n])
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
        if (best_error<tol*max_values) || (n>=max_degree)|| #n>=6 ||
            ((n-best_n>=lookaheaad)&&(best_error<1e-2*max_values))
            break
        end
        if n>=(m>>1) break end

        push!(chosen_index,next_index)
        delete!(wait_index,next_index)
        R[next_index]=0
    end
    
    if isAD

        return best_weight,grid[best_index],values[best_index],best_index
    end

    return best_weight,grid[best_index],values[best_index]

end


#---------------

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

# make tangent mesh
function make_mesh(output_bound::Float64,output_number::Int64)
    p=2.1
    grid=tan.(collect(range(-π/p,π/p,output_number)))
    return grid/grid[end]*output_bound
end





#--------------------------------------
# Aaa algorithm for delta spectral density

function DireInverse_check(β::Float64,N::Int64,γ::Vector{Float64})
    poles=(collect(0:N-1).+0.5)*2π/β
    grid=im*poles
    G_values=generate_G_values_delta(poles,γ,grid)
    return DireInverse(poles,grid,G_values)
end


function generate_G_values_delta(poles::Vector{Float64},γ::Vector{Float64},grid::Vector{ComplexF64})
    @assert length(poles)==length(γ)==length(grid)
    n=length(poles)
    G_values=zeros(ComplexF64,n)
    for i=1:n
        for j=1:n
            G_values[i]+=γ[j]/(grid[i]-poles[j])
        end
    end
    return G_values
end

function DireInverse(poles::Vector{Float64},grid::Vector{ComplexF64},values::Vector{ComplexF64})
    @assert length(poles)==length(values)==length(grid)
    n=length(grid)
    A=Matrix{ComplexF64}(undef,n,n)
    #=
    γ=zeros(ComplexF64,n)
    poles1=copy(poles)
    values1=copy(values)
    grid1=copy(grid)
    for k=1:n
        
        for i=1:n
            for j=1:n
                A[i,j]=1/(grid1[i]-poles1[j])
            end
        end
        γ[k]=(A\values1)[1]
        poles1=permu(poles1)
        grid1=permu(grid1)
        values1=permu(values1)
    end

    return real.(γ)
    =#

    
    #=
    for i=1:n
        for j=1:n
            A[i,j]=1/(grid[i]-poles[j])
        end
    end

    F=qr(A)
    return real.(F\values)
    =#



    γ=zeros(ComplexF64,n)
    for k=1:n
        poles1=permu_closest(poles,k)
        values1=permu_closest(values,k)
        grid1=permu_closest(grid,k)
        for i=1:n
            for j=1:n
                A[i,j]=1/(grid1[i]-poles1[j])
            end
        end
        γ[k]=(A\values1)[1]
    end

    return real.(γ)
    

    
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