$\rm 1.~$
For 
$${\rm Function}:A(x)=\sum \gamma_i\delta(x-x_i)$$

$${\rm Input}:\{\omega_n=(n+\frac{1}{2})\frac{2\pi}{\beta}\},~n=0,..,N-1\\
\{G(i\omega_n)=\int_{R}\frac{A(x)}{i\omega_n-x}dx\}+{\rm noise},~n=0,..,N-1$$

$$~$$
$${\rm Output}: {\rm array}:mesh\\
{\rm Function}:\widetilde{A}(x)$$

In the process to reconstruct the spectral density function, can we just set the poles of of $\widetilde{A}$ as $\{x_i\},~i=0,..,N-1$ ?

In the barycentric method of ACFlow, it use aaa algorithm to get a rational approximation $\frac{N(z)}{D(z)}$ and get poles from it.

But with this method , you can only get $N/2$ poles as most and they can't be accurately $\{i\omega_n\}$.

So can we directly set poles as $\{\omega_n\}$ ?

$\rm 2.~(Done)$
Can AD works for greedy algorithm?

An example for AD doesn't work for greedy algorithm is in `examples/ADforGreedy.jl`

```julia
using Enzyme

function my_greedy(vec::Vector{Float64})
    pick=similar(vec)
    best_pick=similar(vec)
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
    #@show best_n,n,best_error
    return sum(best_pick.^2)
end

function my_max(vec::Vector{Float64})
    return maximum(abs.(vec))
end

N=15
vec=2*rand(N).-1
vec=sign.(vec).*sqrt.(abs.(vec))
my_greedy(vec)




x=2*rand(N)
dx=Vector{Vector{Float64}}(undef,N)
for i=1:N
    dx[i]=fill(0.0,N)
    dx[i][i]=1.0
end
dx=Tuple(dx)

my_greedy(x)

autodiff(ForwardWithPrimal, x->my_max(x), BatchDuplicated(x, dx))
autodiff(ForwardWithPrimal, x->my_greedy(x), BatchDuplicated(x, dx))
```

Solution: replace `fill(0.0,length(vec))` with `similar(vec)` and make program think it a variable.

$$~$$

$\rm 3.~$
Usualy when does aaa algorithm break?

$$~$$

$\rm 4.~(Done)$
How to apply AD for svd ?

$$~$$

$\rm 5.~$
How to get all amplitudes of poles with high accuracy?

$$~$$

$\rm 6.~(Done)$
Why slight noise can break the aaa algorithm greatly?

Because the svd() function ACFlow use will throw $V[:,n+1:m]$ for ${\rm size}(A)=n\times m$ when $m\geq n$

$$~$$

$\rm 7.~$
How to rule AD by self-defined function?.

$$~$$

$\rm 8.~$
How to improve the accuracy when do Unitary decomposition on $$L_{sub}'*L_{sub}$$ 

Sol: Svd on $L_{sub}$ is better than svd on $L_{sub}'*L_{sub}$ and eigen on $L_{sub}'*L_{sub}$.

Optimize may be a good idea.

$$~$$

