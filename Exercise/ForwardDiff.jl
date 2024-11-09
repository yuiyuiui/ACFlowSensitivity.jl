using ForwardDiff
using LinearAlgebra
A=rand(3,3)
function my_func(A::Matrix{Float64})
    _,b,_=svd(A)
    return sum(b)
end
ForwardDiff.gradient(my_func,A)