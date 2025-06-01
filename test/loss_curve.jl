using LinearAlgebra, Random, Test

Random.seed!(3)

function ∂²loss_∂p∂y(p, x, y)
    a, b, c, d = p
    L = length(x)

    # 计算 sigmoid 函数及其相关项
    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    s1 = s .* (1 .- s)  # s1 = s * (1 - s)

    # 初始化混合偏导数矩阵
    ∂²loss_∂p∂y_matrix = zeros(4, L)

    # 填充矩阵
    ∂²loss_∂p∂y_matrix[1, :] .= -2  # ∂²loss/∂a∂y_i
    ∂²loss_∂p∂y_matrix[2, :] = -2 * s  # ∂²loss/∂b∂y_i
    ∂²loss_∂p∂y_matrix[3, :] = 2 * b * d * s1  # ∂²loss/∂c∂y_i
    ∂²loss_∂p∂y_matrix[4, :] = -2 * b * s1 .* (x .- c)  # ∂²loss/∂d∂y_i

    return ∂²loss_∂p∂y_matrix
end

function finite_difference_∂²loss_∂p∂y(p, x, y, ϵ=1e-6)
    L = length(x)
    ∂²loss_∂p∂y_matrix = zeros(4, L)

    for i in 1:4
        p_plus = copy(p)
        p_minus = copy(p)
        p_plus[i] += ϵ
        p_minus[i] -= ϵ

        ∂loss_plus = ∂loss_∂y(p_plus, x, y)
        ∂loss_minus = ∂loss_∂y(p_minus, x, y)

        ∂²loss_∂p∂y_matrix[i, :] = (∂loss_plus - ∂loss_minus) / (2 * ϵ)
    end

    return ∂²loss_∂p∂y_matrix
end

function ∂loss_∂y(p, x, y)
    a, b, c, d = p
    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    return -2 * (a .+ b * s .- y)
end

function loss(p, x, y)
    a, b, c, d=p
    s=1 ./ (1 .+ exp.(-d*(x .- c)))
    r=a .+ b*s - y
    return sum(r .^ 2)
end

function ∂²loss_curveDiv∂p²(p, x, y)
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
    Jcd = -2 *
          b *
          sum(s1 .* (b * d * s1 .* (x .- c) .+ (1 .+ d * (x .- c) .* (1 .- 2 * s)) .* r))

    return [Jaa Jab Jac Jad; Jab Jbb Jbc Jbd; Jac Jbc Jcc Jcd; Jad Jbd Jcd Jdd]
end

function ∂loss_curveDiv∂p(p, x, y)
    a, b, c, d=p
    s=1 ./ (1 .+ exp.(-d*(x .- c)))
    r=a .+ b*s - y

    Ja=2*sum(r)
    Jb=2*sum(s .* r)
    Jc=-2*b*d*sum(s .* (1 .- s) .* r)
    Jd=2*b*sum(s .* (1 .- s) .* (x .- c) .* r)
    return [Ja, Jb, Jc, Jd]
end

@testset "test ∂²loss_∂p∂y" begin
    # 测试
    p = [1.0, 2.0, 0.5, 0.1]
    x = rand(100)
    y = rand(100)

    analytic_grad = ∂²loss_∂p∂y(p, x, y)
    finite_diff_grad = finite_difference_∂²loss_∂p∂y(p, x, y)

    @test norm(analytic_grad - finite_diff_grad) < 1e-6
end

# 有限差分法验证二阶导数
function finite_difference_∂²loss_∂p²(p, x, y, ϵ=1e-6)
    n = length(p)
    H = zeros(n, n)  # 初始化 Hessian 矩阵

    for i in 1:n
        for j in 1:n
            # 单位向量
            ei = zeros(n)
            ej = zeros(n)
            ei[i] = 1
            ej[j] = 1

            # 有限差分公式
            H[i, j] = (loss(p + ϵ * ei + ϵ * ej, x, y) - loss(p + ϵ * ei, x, y) -
                       loss(p + ϵ * ej, x, y) + loss(p, x, y)) / ϵ^2
        end
    end

    return H
end

# 测试
p = [1.0, 2.0, 0.5, 0.1]  # 参数
x = rand(100)             # 输入数据
y = rand(100)             # 目标值

# 解析解
analytic_H = ∂²loss_curveDiv∂p²(p, x, y)

# 有限差分近似
finite_diff_H = finite_difference_∂²loss_∂p²(p, x, y)

# 计算相对误差
relative_error = norm(analytic_H - finite_diff_H) / norm(analytic_H)
println("Relative Error: ", relative_error)

a = 12:-1:-3
a = collect(a) .+ 0.0

b = rand(16)

p = [1.0, 2.0, 0.5, 0.1]

∂²loss_curveDiv∂p²(p, a, b)
