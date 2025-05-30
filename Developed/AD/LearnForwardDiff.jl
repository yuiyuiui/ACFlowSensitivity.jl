using ForwardDiff
using Zygote

Zygote.gradient(x->x^2, 1.0)

ForwardDiff.gradient(x->x[1]^2, [1.0])

using ForwardDiff

# 1. 定义一个自定义 sin 运算符
import Base: sin

# 重载 sin 运算符，改变其导数传播规则
sin(a::ForwardDiff.Dual) = ForwardDiff.Dual(sin(a.value), cos(a.value) * a.perturbation)

# 2. 定义目标函数
function my_function(x)
    return sin(x[1]) # 目标函数是 sin(x)
end

# 3. 计算目标函数的梯度
x = [2.0]
grad = ForwardDiff.gradient(my_function, x)

println("sin(x) 的值: ", sin(x))  # 输出 sin(x) 的值
println("sin(x) 的导数: ", grad)  # 输出 sin(x) 的导数（梯度部分）
