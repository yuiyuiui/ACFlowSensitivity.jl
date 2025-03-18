using LinearAlgebra,Random,Test

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

@testset "test ∂²loss_∂p∂y" begin
# 测试
p = [1.0, 2.0, 0.5, 0.1]
x = rand(100)
y = rand(100)

    analytic_grad = ∂²loss_∂p∂y(p, x, y)
    finite_diff_grad = finite_difference_∂²loss_∂p∂y(p, x, y)

    @test norm(analytic_grad - finite_diff_grad) < 1e-6
end
