tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

function integral(f::Function, a::T, b::T; h::T=T(1e-4)) where {T<:Real}
    n_raw = floor((b - a) / h)
    n = Int(n_raw)
    if isodd(n)
        n -= 1
    end
    if n < 2
        error("step is too large")
    end

    fa = f(a)
    !(typeof(fa) <: Union{T,Complex{T}}) &&
        error("Type of the output of f should be consistent with its input")
    fb = f(a + h * T(n))
    acc = fa + fb

    @inbounds for i in 1:(n - 1)
        x = a + h * T(i)
        coeff = isodd(i) ? T(4) : T(2)
        acc += coeff * f(x)
    end

    return acc * (h / T(3))
end

function Lp(f::Function, p::Real, a::T, b::T; h::T=T(1e-4)) where {T<:Real}
    Tp = T(p)
    return integral(x->abs(f(x))^Tp, a, b; h=h)^(1/Tp)
end

#-----------------------------------------

#= for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0], [ε], [1 / (sqrt(2π) * ε)])
end
=#

# Newton Method
function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
    resid = nothing
    step = T(1)
    limit = eps(T)^(1//4)
    try
        resid = -pinv(J) * f
    catch
        resid = zeros(T, length(feed))
    end
    if any(x -> x > limit, abs.(feed))
        ratio = abs.(resid ./ feed)
        max_ratio = maximum(ratio[abs.(feed) .> limit])
        if max_ratio > T(1)
            step = T(1) / max_ratio
        end
    end
    return feed + step .* resid
end
function newton(fun::Function, grad::Function, guess::Vector{T}; maxiter::Int=20000,tol::T=T(1e-5)) where T<:Real
    counter = 0
    feed = copy(guess)
    f = fun(feed)
    J = grad(feed)
    back = _apply(feed, f, J)
    reach_tol = false

    while true
        counter = counter + 1
        feed += T(1//2) * (back - feed)

        f = fun(feed)
        J = grad(feed)
        back = _apply(feed, f, J)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum(abs.(back - feed)) < tol
            break
        end
    end

    if counter > maxiter
        println("Tolerance is reached in newton()!")
        @show norm(grad(back))
        reach_tol = true
    end

    return back, counter, reach_tol
end

#= Curve fitting for ϕ(x;a,b,c,d)=a+b/(1+exp(-d(x-c)))
function loss(p,x,y)
    a,b,c,d=p
    s=1 ./ ( 1 .+ exp.(-d*(x.-c))  )
    r=a .+ b*s - y
    return sum(r.^2)
end
=#
function _∂loss_curveDiv∂p(p::Vector{T},x::Vector{T},y::Vector{T}) where T<:Real
    a, b, c, d=p
    s=1 ./ (1 .+ exp.(-d*(x .- c)))
    r=a .+ b*s - y

    Ja=2*sum(r)
    Jb=2*sum(s .* r)
    Jc=-2*b*d*sum(s .* (1 .- s) .* r)
    Jd=2*b*sum(s .* (1 .- s) .* (x .- c) .* r)
    return [Ja, Jb, Jc, Jd]
end
function _∂²loss_curveDiv∂p²(p::Vector{T},x::Vector{T},y::Vector{T}) where T<:Real
    a, b, c, d = p
    L = length(x)

    # 计算 sigmoid 函数及其相关项
    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    s1 = s .* (1 .- s)  # s1 = s * (1 - s)
    r = a .+ b * s .- y  # 残差项

    # 填充对角元素
    Jaa = 2 * L
    Jbb = 2 * sum(s .^ 2)
    Jcc = 2 * b^2 * d^2 * sum(s .^ 2 .* (1 .- s) .^ 2) +
          2 * b * d^2 * sum(s1 .* (1 .- 2 * s) .* r)
    Jdd = 2 * sum(b^2 * s .^ 2 .* (1 .- s) .^ 2 .* (x .- c) .^ 2 +
                  b * (x .- c) .^ 2 .* s1 .* (1 .- 2 * s) .* r)

    # 填充非对角元素
    Jab = 2 * sum(s)
    Jac = -2 * b * d * sum(s1)
    Jad = 2 * b * sum(s1 .* (x .- c))
    Jbc = -2 * d * sum(s1 .* (b * s .+ r))
    Jbd = 2 * sum(s1 .* (x .- c) .* (b * s .+ r))
    Jcd = -2 * b * sum(s1 .* (b * d * s1 .* (x .- c) .+ (1 .+ d * (x .- c) .* (1 .- 2 * s)) .* r))
    return [Jaa Jab Jac Jad; Jab Jbb Jbc Jbd; Jac Jbc Jcc Jcd; Jad Jbd Jcd Jdd]
end
function curve_fit(x::Vector{T}, y::Vector{T}; guess::Vector{T}=[T(0),T(5),T(2),T(0)], maxiter::Int=20000,tol::T=T(1e-5)) where T<:Real
    @assert length(x)==length(y)
    return newton(p->_∂loss_curveDiv∂p(p,x,y), p->_∂²loss_curveDiv∂p²(p,x,y), guess; maxiter=maxiter, tol=tol)
end
