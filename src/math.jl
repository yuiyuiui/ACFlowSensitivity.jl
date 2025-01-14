# for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0],[ε],[1/(sqrt(2π)*ε)])
end

# Why we don't use quadgk, but accomplish it by ourselves ?
# Because it takes too much time to calculate the first gradient( parameter->quadgk( func_type(parameter),int_low,int_up )[1], parameter )
# after "using QuadGK"
function Lp_norm(f,p::Real;int_low::Float64=-8.0,int_up::Float64=8.0,step::Float64=1e-4)
    int_field=collect(int_low:step:int_up)
    n=length(int_field)
    values=abs.(f.(int_field)).^p
    values1=view(values,1:n-1)
    values2=view(values,2: n)
    return sum(  ( values1+values2)*step/2  )^(1/p)
end


function my_BFGS(f, grad, x0; tol=1e-6, max_iter=2000)
    x = x0
    n = length(x)
    H = I(n)  # 初始Hessian矩阵的逆近似

    for iter in 1:max_iter
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

function linesearch(f, grad, x, p; c1=1e-4, c2=0.9, max_iter=20)
    α = 1.0
    for iter in 1:max_iter
        if f(x + α * p) <= f(x) + c1 * α * dot(grad(x), p) &&
           dot(grad(x + α * p), p) >= c2 * dot(grad(x), p)
            return α
        end
        α /= 2
    end
    return α
end

#= 自己写的梯度下降法
function my_GD_v1(f, grad, x0; tol=1e-6, max_iter=2000)
    res=copy(x0)
    ite=0
    while true
        ratio=norm(grad(res))
        # 归一化方向
        direct=-grad(res)/ratio
        if ratio<tol || ite>=max_iter
            println("iterations is $ite")
            return res
        end

        f_now=f(res)
        step=1.0
        while f(res+direct*step)<f_now-step*ratio*2/3
            step*=2
        end
        while f(res+direct*step)>=f_now-step*ratio/3
            step/=2
        end
        res=res+direct*step
        ite+=1
    end
end
=#
