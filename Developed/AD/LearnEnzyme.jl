using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

function f(y, x)
    y .= x .^ 2
    return sum(y)
end

function EnzymeRules.forward(
    config::FwdConfig,
    func::Const{typeof(f)},
    ::Type{<:Duplicated},
    y::Duplicated,
    x::Duplicated,
)
    println("Using custom rule!")
    ret = func.val(y.val, x.val)
    y.dval .= 2 .* x.val .* x.dval
    return Duplicated(ret, sum(y.dval))
end

x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2# function to differentiate

@show autodiff(ForwardWithPrimal, f, Duplicated(y, dy), Duplicated(x, dx))
@show y, dy
@show autodiff(ForwardWithPrimal, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1] when g is run

#-----------------------------------------------------------------------------------------------------------------------------------


my_func(x::Vector{ComplexF64}) = sum(abs.(x))

x = [1.0+0.0im, 2.0+0.0im, 3.0+0.0im]
dx = [1.0+0.0im, 0.0+0.0im, 0.0+0.0im]

@show autodiff(Forward, my_func, Duplicated(x, dx))


#-----------------------------------------------------------------------------------------------------------------------------------


function ll(y::Vector{Float64}, x::Vector{Float64})
    k=[2.0, 1.0]
    b=[1.0, 0.0]
    y .= k .* x .+ b
    return sum(y)
end

x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]
@show ll(y, x)


function EnzymeRules.forward(
    config::FwdConfig,
    func::Const{typeof(ll)},
    ::Type{<:Duplicated},
    y::Duplicated,
    x::Duplicated,
)
    ret = func.val(y.val, x.val)
    y.dval .= [2.0, 1.0] .* x.dval
    return Duplicated(ret, sum(y.dval))
end

@show autodiff(ForwardWithPrimal, ll, Duplicated(y, dy), Duplicated(x, dx))


#-----------------------------------------------------------------------------------------------------------------------------------


struct Line<:Function
    k::Vector{Float64}
    b::Vector{Float64}
end

function (l::Line)(y::Vector{Float64}, x::Vector{Float64})
    y .= l.k .* x .+ l.b
    return sum(y)
end

l = Line([2.0, 1.0], [1.0, 0.0])
@show l(y, x)

function EnzymeRules.forward(
    config::FwdConfig,
    func::Const{<:Line},
    ::Type{<:Duplicated},
    y::Duplicated,
    x::Duplicated,
)
    @show func.val.k
    ret = func.val(y.val, x.val)
    y.dval .= func.val.k .* x.dval
    return Duplicated(ret, sum(y.dval))
end

x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

@show autodiff(ForwardWithPrimal, Const(l), Duplicated(y, dy), Duplicated(x, dx))



#-----------------------------------------------------------------------------------------------------------------------------------




function my_func(x::Vector{Float64})
    return sum(x)
end

function EnzymeRules.forward(
    config::FwdConfig,
    func::Const{typeof(my_func)},
    ::Type{<:Duplicated},
    y::Duplicated,
    x::Duplicated,
)
    y.dval .= 1.0
    return Duplicated(func.val(x.val), length(x.val))
end

@show autodiff(ForwardWithPrimal, my_func, Duplicated(y, dy), Duplicated(x, dx))





function ff(x)
    return [x, x+1]
end
x=1.0
dx=1.0

@show autodiff(Forward, ff, Duplicated(x, dx))[1]


function my_svd(A::Matrix{ComplexF64})
    return sum(A)
end

A=rand(ComplexF64, 2, 2)
dA=[1.0+0im 0; 0 0]

@show autodiff(ForwardWithPrimal, my_svd, Duplicated(A, dA))

#-----------------------------------------------------------------------------------------------------------------------------------

struct my_Line<:Function
    k::Vector{Float64}
    b::Vector{Float64}
end

function (l::my_Line)(x::Vector{Float64})
    return sum(l.k .* x .+ l.b)
end

l=my_Line([2.0, 1.0], [1.0, 0.0])
x=[1.0, 2.0]
dx=[1.0, 0.0]

@show l(x)

@show autodiff(ForwardWithPrimal, Const(l), Duplicated(x, dx))

function EnzymeRules.forward(
    config::FwdConfig,
    func::Const{<:my_Line},
    ::Type{<:Duplicated},
    x::Duplicated,
)
    ret = func.val(x.val)
    return Duplicated(ret, sum(func.val.k .* x.dval))
end

@show autodiff(ForwardWithPrimal, Const(l), Duplicated(x, dx))

function my_func(x::Vector{ComplexF64})
    return sum(abs.(x) .^ 2)
end

x=[1.0+2.0im, 2.0+0.0im]
dx=[1.0-1.0im, 0.0+0.0im]
dx1=[1.0-1.0im, 0.0+0.0im]

function EnzymeRules.forward(
    config::FwdConfig,
    func::Const{typeof(my_func)},
    ::Type{<:Duplicated},
    x::Duplicated,
)
    ret = func.val(x.val)
    dval=2*real.(x.val) .* real.(x.dval)+im*2*imag.(x.val) .* imag.(x.dval)
    return Duplicated(ret, dval)
end

@show autodiff(ForwardWithPrimal, my_func, Duplicated(x, dx))
@show autodiff(ForwardWithPrimal, my_func, Duplicated(x, dx1))


using Zygote
@show Zygote.gradient(my_func, x)


#-----------------------------------------------------------------------------------------------------------------------------------
f(z) = z * z

# a fixed input to use for testing
z = 1.0+2.0im

grad_u = Enzyme.autodiff(Reverse, z->real(f(z)), Active, Active(z))[1][1]
grad_v = Enzyme.autodiff(Reverse, z->imag(f(z)), Active, Active(z))[1][1]

(grad_u, grad_v)
# output
(6.2 - 5.4im, 5.4 + 6.2im)
