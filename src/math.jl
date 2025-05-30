abstract type Method end

# 定义具体的方法类型
struct Newton <: Method end
struct GD <: Method end





#-----------------------------------------

# for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0], [ε], [1 / (sqrt(2π) * ε)])
end

# Why we don't use quadgk, but accomplish it by ourselves ?
# Because it takes too much time to calculate the first gradient( parameter->quadgk( func_type(parameter),int_low,int_up )[1], parameter )
# after "using QuadGK"
function Lp_norm(
    f,
    p::Real;
    int_low::Float64 = -8.0,
    int_up::Float64 = 8.0,
    step::Float64 = 1e-4,
)
    int_field = collect(int_low:step:int_up)
    n = length(int_field)
    values = abs.(f.(int_field)) .^ p
    values1 = view(values, 1:(n-1))
    values2 = view(values, 2:n)
    return sum((values1 + values2) * step / 2)^(1 / p)
end


function my_BFGS(f, grad, x0; tol = 1e-6, max_iter = 2000)
    x = x0
    n = length(x)
    H = I(n)  # 初始Hessian矩阵的逆近似

    for iter = 1:max_iter
        g = grad(x)
        if norm(g) < tol
            println("Converged in $iter iterations")
            return x
        end

        p = -H * g
        α = linesearch(f, grad, x, p)

        s = α * p
        x_new = x + s
        g_new = grad(x_new)

        y = g_new - g
        rho = 1 / dot(y, s)
        H = (I(n) - rho * s * y') * H * (I(n) - rho * y * s') + rho * s * s'

        x = x_new
    end

    println("Maximum iterations reached")
    return x
end

function linesearch(f, grad, x, p; c1 = 1e-4, c2 = 0.9, max_iter = 20)
    α = 1.0
    for iter = 1:max_iter
        if f(x + α * p) <= f(x) + c1 * α * dot(grad(x), p) &&
           dot(grad(x + α * p), p) >= c2 * dot(grad(x), p)
            return α
        end
        α /= 2
    end
    return α
end

# 自己写的梯度下降法
function my_GD_v1(f, grad, x0; tol = 1e-6, max_iter = 20000)
    res = copy(x0)
    ite = 0
    while true
        ratio = norm(grad(res))
        # 归一化方向
        direct = -grad(res) / ratio
        if ratio < tol || ite >= max_iter
            println("Iterations is $ite, ratio is $ratio")
            return res
        end

        f_now = f(res)
        step = 1.0
        while f(res + direct * step) < f_now - step * ratio * 2 / 3
            step *= 2
        end
        while f(res + direct * step) >= f_now - step * ratio / 3
            step /= 2
        end
        res = res + direct * step
        ite += 1
    end
end


# 自己写的梯度下降法，斜率的角平分线版本，v2效果一般会比v1好一些
function my_GD_v2(f, grad, x0; tol = 1e-4, max_iter = 20000)
    res = copy(x0)
    ite = 0
    reach_tol = false
    while true
        ratio = norm(grad(res))
        # 归一化方向
        direct = -grad(res) / ratio
        if ratio < tol || ite >= max_iter
            if ite >= max_iter
                reach_tol = true
                println("Tolerance is reached in GD()!")
            end
            return res, ite, reach_tol
        end

        f_now = f(res)
        step = 1.0
        mid_div = (sqrt(ratio^2 + 1) - 1) / ratio
        while f(res + direct * step) < f_now - step * mid_div
            step *= 2
        end
        while f(res + direct * step) >= f_now - step * mid_div
            step /= 2
        end
        list = collect(range(0, 2 * step, 20)[2:end])
        min_value = f_now
        for i ∈ 1:19
            if f(res + direct * list[i]) < min_value
                step = list[i]
                min_value = f(res + direct * list[i])
            end
        end


        res = res + direct * step
        ite += 1
    end
end


# 牛顿法，ACFlow 的步长处理
function my_newton(
    fun::Function,
    grad::Function,
    guess;
    maxiter::Int64 = 20000,
    mixing::Float64 = 0.5,
)
    function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = 1.0
        limit = 1e-4
        try
            resid = -pinv(J) * f
        catch
            resid = zeros(Float64, length(feed))
        end
        if any(x -> x > limit, abs.(feed))
            ratio = abs.(resid ./ feed)
            max_ratio = maximum(ratio[abs.(feed) .> limit])
            if max_ratio > 1.0
                step = 1.0 / max_ratio
            end
        end
        return feed + step .* resid
    end

    counter = 0
    feed = copy(guess)
    f = fun(feed)
    J = grad(feed)
    back = _apply(feed, f, J)
    reach_tol = false

    while true
        counter = counter + 1
        feed += mixing * (back - feed)

        f = fun(feed)
        J = grad(feed)
        back = _apply(feed, f, J)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum(abs.(back - feed)) < 1e-5
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




# ϕ(x;a,b,c,d)=a+b/(1+exp(-d(x-c))) 的曲线拟合
#=
function loss(p,x,y)
    a,b,c,d=p
    s=1 ./ ( 1 .+ exp.(-d*(x.-c))  )
    r=a .+ b*s - y
    return sum(r.^2)
end
=#
function my_curve_fit(
    xx::Vector{Float64},
    yy::Vector{Float64},
    guess::Vector{Float64},
    method::GD,
)
    @assert length(xx)==length(yy)
    loss(p) = sum((p[1] .+ p[2] ./ (1 .+ exp.(-p[4] * (xx .- p[3]))) - yy) .^ 2)
    L=length(xx)

    # 公式的正确性已被验证
    function J(p)
        @assert length(p) == 4
        a, b, c, d = p
        mid1 = 2 * (a .+ b ./ (1 .+ exp.(-d * (xx .- c))) - yy)
        mid2 = 1 ./ (1 .+ exp.(-d * (xx .- c)))
        mid3 = mid2 .^ 2 * (-b) .* exp.(-d * (xx .- c))
        ∂a = mid1' * ones(L)
        ∂b = mid1' * mid2
        ∂c = mid1' * mid3 * d
        ∂d = mid1' * (mid3 .* (c .- xx))
        return [∂a, ∂b, ∂c, ∂d]
    end

    return my_GD_v2(loss, J, guess)
end


function my_curve_fit(
    x::Vector{Float64},
    y::Vector{Float64},
    guess::Vector{Float64},
    method::Newton,
)
    @assert length(x)==length(y)
    L=length(x)

    function _∂loss_curveDiv∂p(p)
        a, b, c, d=p
        s=1 ./ (1 .+ exp.(-d*(x .- c)))
        r=a .+ b*s - y

        Ja=2*sum(r)
        Jb=2*sum(s .* r)
        Jc=-2*b*d*sum(s .* (1 .- s) .* r)
        Jd=2*b*sum(s .* (1 .- s) .* (x .- c) .* r)
        return [Ja, Jb, Jc, Jd]
    end

    function _∂²loss_curveDiv∂p²(p)
        a, b, c, d = p
        L = length(x)

        # 计算 sigmoid 函数及其相关项
        s = 1 ./ (1 .+ exp.(-d * (x .- c)))
        s1 = s .* (1 .- s)  # s1 = s * (1 - s)
        r = a .+ b * s .- y  # 残差项

        # 填充对角元素
        Jaa = 2 * L
        Jbb = 2 * sum(s .^ 2)
        Jcc =
            2 * b^2 * d^2 * sum(s .^ 2 .* (1 .- s) .^ 2) +
            2 * b * d^2 * sum(s1 .* (1 .- 2 * s) .* r)
        Jdd =
            2 * sum(
                b^2 * s .^ 2 .* (1 .- s) .^ 2 .* (x .- c) .^ 2 +
                b * (x .- c) .^ 2 .* s1 .* (1 .- 2 * s) .* r,
            )

        # 填充非对角元素
        Jab = 2 * sum(s)
        Jac = -2 * b * d * sum(s1)
        Jad = 2 * b * sum(s1 .* (x .- c))
        Jbc = -2 * d * sum(s1 .* (b * s .+ r))
        Jbd = 2 * sum(s1 .* (x .- c) .* (b * s .+ r))
        Jcd =
            -2 *
            b *
            sum(s1 .* (b * d * s1 .* (x .- c) .+ (1 .+ d * (x .- c) .* (1 .- 2 * s)) .* r))

        return [Jaa Jab Jac Jad; Jab Jbb Jbc Jbd; Jac Jbc Jcc Jcd; Jad Jbd Jcd Jdd]
    end

    return my_newton(_∂loss_curveDiv∂p, _∂²loss_curveDiv∂p², guess)

end
