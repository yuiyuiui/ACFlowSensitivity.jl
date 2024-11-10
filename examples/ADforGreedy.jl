using Enzyme


# The perturbation does not affect the choice.
function my_greedy(vec::Vector{Float64})
    pick=fill(0.0,length(vec))
    best_pick=fill(0.0,length(vec))
    wait_set=Set(vec)
    n=0
    partsum=0.0
    best_error=Inf
    best_n=0
    while n<length(vec)
        wait_vec=collect(wait_set)
        error,p=findmin( abs.( (wait_vec.+partsum)/(n+1) ) )
        n+=1
        pick[n]=wait_vec[p]
        wait_set=setdiff(wait_set,pick[n])
        partsum+=pick[n]
        if error<best_error
            best_error=error
            best_pick=pick
            best_n=n
        end
        if best_error<1e-2
            break
        end
    end
    @show best_n,n,best_error
    return sum(abs.(best_pick))
end

function my_max(vec::Vector{Float64})
    return maximum(abs.(vec))
end

N=15
vec=2*rand(N).-1
vec=sign.(vec).*sqrt.(abs.(vec))
my_greedy(vec)


x=2*rand(N).-1
dx=Vector{Vector{Float64}}(undef,N)
for i=1:N
    dx[i]=fill(0.0,N)
    dx[i][i]=1.0
end
dx=Tuple(dx);

my_greedy(x)

autodiff(ForwardWithPrimal, x->my_max(x), BatchDuplicated(x, dx))
autodiff(ForwardWithPrimal, x->my_greedy(x), BatchDuplicated(x, dx))


# The perturbation affects the choice.
