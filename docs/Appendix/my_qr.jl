using LinearAlgebra

function householder_qr(A)
    m, n = size(A)
    @assert m >= n
    @assert det(A'*A) != 0
    Q = Matrix{eltype(A)}(I, m, m)  # 初始化 Q 为单位矩阵
    R = copy(A)
    for k = 1:n
        a=R[k:end, k]
        e1=zeros(eltype(A), length(a))
        e1[1]=norm(a)
        v=a-e1
        v/=norm(v)
        Hk=I(length(v))-2*(v*v')
        R[k:end, k:end] = Hk * R[k:end, k:end]
        Q[:, k:end] = Q[:, k:end] * Hk
    end

    return Q[:, 1:n], R[1:n, 1:n]  # 返回 Q 和 R
end

# 测试
A=rand(5, 2)
Q, R = householder_qr(A)
println("Q = ", Q)
println("R = ", R)
println("Check A = Q * R: ", Q * R ≈ A)
