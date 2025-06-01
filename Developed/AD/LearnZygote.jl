using Zygote
using Zygote: @adjoint

gradient(x->3x^2+2x+1, 5)

W=rand(2, 3);
x=rand(3)
gradient(W->sum(W*x), W)[1]

function my_func1(Z::Matrix{ComplexF64})
    return sum(abs.(Z) .^ 2)
end

my_matrix=[1.0 2.0; 3.0 4.0] .+ 0.0im
gradient(Z->my_func1(Z), my_matrix)

c=[1, 2]
gradient(c->c[1]*c[2], c)

d=Dict()

gradient(5) do x
    d[:x]=x
    return d[:x]*d[:x]
end

linear(θ, x)=θ[:W]*x .+ θ[:b]
x=rand(5)
θ=Dict(:W=>rand(2, 5), :b=>rand(2))

θ̄ = gradient(θ -> sum(linear(θ, x)), θ)[1]

struct Linear
    W::AbstractMatrix{Float64}
    b::AbstractVector{Float64}
end

(l::Linear)(x)=l.W*x .+ l.b

model=Linear(rand(2, 5), rand(2))

dmodel = gradient(model -> sum(model(x)), model)[1]

l=Linear(rand(2, 3), rand(2))

l(rand(3))

x=[1 2 3; 4 5 6];
y=[7, 8];
z=[1, 10, 100]

g=gradient(Params([x, y])) do
    return sum(x .* y .* z')
end

x=ones(5)

W=rand(2, 5);
b=rand(2)
linear(x)=W*x .+ b
grads=gradient(()->sum(linear(x)), Params([W, b]))

grads[W]
grads[b]

jclock(x) = ccall(:clock, Int32, ()) * x
jclock(1.0)

function ChainRulesCore.rrule(::typeof(jclock), x)
    y=jclock(x)
    jc_pullback(dy) = (ChainRulesCore.NoTangent(), dy*y)
    return y, jc_pullback
end

gradient(jclock, rand())

struct my_type
    x::Float64
    y::Float64
end

# Zygote.pullback(f,x0) return a value f(x0) and a afunction f'
y, back=Zygote.pullback(sin, π/3)

y
sqrt(3)/2
back(1.0)
back

#--------------------------------
# Adjoint
# Adjoint 实际上定义的是反向AD计算过程中，某一层函数映射的反向传播的计算过程
# 也就是说，@adjoint f 指的是对g(f(x))，要计算d(g(f))/dx，已经知道d(g(f))/df，则
# d(g(f))/dx=d(g(f))/df*d(f)/dx 如何计算

mul(a, b)=a*b
@adjoint mul(a, b) = mul(a, b), c->(c*b, a*c)

gradient(mul, 2, 3)

import Base: +, -
struct Point
    x::Float64
    y::Float64
end

width(p::Point)=p.x
height(p::Point)=p.y

a::Point+b::Point=Point(width(a)+width(b), height(a)+height(b))
a::Point-b::Point=Point(width(a)-width(b), height(a)-height(b))
dist(p::Point)=sqrt(width(p)^2+height(p)^2)

gradient(a->dist(a), Point(1, 2))[1]

# d(P(a,b))/dP=Point(.,.)
# The partial chain reach until Point, but not the real pair
@adjoint width(p::Point)=p.x, dp->(Point(dp, 0.0),)
@adjoint height(p::Point)=p.y, dp->(Point(0.0, dp),)

gradient(a->height(a), Point(1, 2))
gradient(a->dist(a), Point(1, 2))[1]

#Reach the real pair
@adjoint Point(a, b) = Point(a, b), p -> (p.x, p.y)
gradient(x->dist(Point(x, 1)), 1)

# x->(nothing, f(x)) means don't get the gradient of the first argument
# nothing is usually used for the argument that is not needed to get gradient
hook(f, x)=x
@adjoint hook(f, x)=x, x->(nothing, f(x))

gradient((a, b) -> hook(ā -> @show(ā), a)*b, 2, 3)

checkpoint(f, x)=f(x)
@adjoint checkpoint(f, x)=f(x), ȳ->Zygote._pullback(f, x)[2](ȳ)
gradient(x->checkpoint(sin, x), π/2)

foo(x)=(println(x); sin(x))
gradient(x->checkpoint(foo, x), 1)

isderiving=false
@adjoint isderiving()=true, _ -> nothing

nestlevel()=0
@adjoint nestlevel()=nestlevel()+1, _->nothing

function f(x)
    println(nestlevel(), " level of nesting")
    return x
end

grad(f, x)=gradient(f, x)[1]

f(1)
grad(f, 1)
grad(x->x*grad(f, x), 1)

gradient(z->abs(z[1])^2+abs(z[2]), [1+im 1-im])[1]

# view jacobian function

function my_func2(x::Matrix{Float64})
    return [x[1]+x[2] x[1]-x[2]; x[1]*x[2] x[1]/x[2]]
end

x=rand(2, 2)
jacobian(my_func2, x)[1]
