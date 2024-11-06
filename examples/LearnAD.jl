#----------
# check efficiency of AD
using ACFlowSensitivity
using Enzyme
using LinearAlgebra



function my_max(vec::Vector{Float64})
    return maximum(abs.(vec))
end

function my_abs(x::Vector{Float64})
    return sum(abs.(x))
end

function my_func1(x)
    if x[1]<=0
        return x[1]
    else
        return x[1]+1
    end
end


function my_minsvd(vec::Vector{Float64})
    vec=sort(vec)
    n=length(vec)
    A=Matrix{Float64}(undef,n,n)
    @show n,size(A)
    for k=1:n
        for j=1:n
            if k==j
                A[k,j]=0
            else
                A[k,j]=1/(vec[k]-vec[j])
            end
        end
    end

    _,_,V=svd(A)
    return norm(V[:,end])
end

function my_reverse_minsvd(x::Array{Float64}, y::Array{Float64})
    y[1]=my_minsvd(x)
    return nothing
end


N=3
x=2*rand(N)
dx=Vector{Vector{Float64}}(undef,N)
for i=1:N
    dx[i]=fill(0.0,N)
    dx[i][i]=1.0
end
dx=Tuple(dx)

my_minsvd(x)

autodiff(ForwardWithPrimal, x->my_minsvd(x), BatchDuplicated(x, dx))


bx = [0.0, 0.0]
y  = [0.0]
by = [1.0];
Enzyme.autodiff(Reverse, my_reverse_minsvd, Duplicated(x, bx), Duplicated(y, by));







