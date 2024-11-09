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




