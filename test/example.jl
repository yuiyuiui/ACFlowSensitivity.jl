using LinearAlgebra, Zygote

function slicerightmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    A[:, :, p] .= A[:, :, p] * B
    return A
end

function slicerightmul!_back(A::Array{T,3}, B::Matrix{T}, p::Int, dC::Array{T,3}) where {T}
    dA = deepcopy(dC)
    dA[:, :, p] .= dC[:, :, p] * B'
    dB = A[:, :, p]' * dC[:, :, p]
    return dA, dB
end

Zygote.@adjoint function slicerightmul!(A::Array{T,3}, B::Matrix{T}, p::Int) where {T}
    C = slicerightmul!(A, B, p)
    function pullback(dC)
        dA, dB = slicerightmul!_back(A, B, p, dC)
        return (dA, dB, nothing)
    end
    return C, pullback
end

function f(grid::Vector{Complex{T}}, Gᵥ::Vector{Complex{T}}) where {T<:Real}
    ngrid = length(grid)
    CT = Complex{T}

    # Allocate memory
    Φ = CT[]
    𝑔 = grid * im

    # Initialize the `abcd` matrix
    𝒜 = [i == j ? CT(1) : CT(0) for i in 1:2, j in 1:2, k in 1:ngrid]

    # Evaluate Φ using recursive algorithm
    Φ = vcat(Φ, Gᵥ[1])
    for j in 1:(ngrid - 1)
        for k in (j + 1):ngrid
            ∏11 = (𝑔[k] - 𝑔[j]) / (𝑔[k] - conj(𝑔[j]))
            ∏12 = Φ[j]
            ∏21 = conj(Φ[j]) * ∏11
            ∏22 = one(CT)
            M = [∏11 ∏12; ∏21 ∏22]
            𝒜 = slicerightmul!(𝒜, M, k)
        end
        num = 𝒜[1, 2, j + 1] - 𝒜[2, 2, j + 1] * Gᵥ[j + 1]
        den = 𝒜[2, 1, j + 1] * Gᵥ[j + 1] - 𝒜[1, 1, j + 1]
        Φ = vcat(Φ, num / den)
    end

    return Φ
end

N = 10
T = Float64
grid = T.(1:N) .+ 0.0im
Gᵥ = rand(Complex{T}, N)
Φ = f(grid, Gᵥ)
gradient(G -> sum(abs2.(f(grid, G))), Gᵥ)

using LinearAlgebra, Zygote
v = rand(3)
function tf(u)
    u = reverse(u)
    return u
end
tf(v)
Zygote.jacobian(tf, v)

using Zygote, LinearAlgebra
function calc_abcd(grid::Vector{Complex{T}}, mesh::Vector{Complex{T}},
                   Φ::Vector{Complex{T}}, alg::NAC) where {T<:Real}
    CT = Complex{T}
    eta = CT(alg.eta)
    ngrid = length(grid)
    nmesh = length(mesh)
    𝑔 = grid * im
    𝑚 = mesh .+ im * eta
    𝒜 = zeros(CT, 2, 2, nmesh)

    function _calc_abcd(𝑧)
        result = Matrix{CT}(I, 2, 2)
        for j in 1:ngrid
            ∏11 = (𝑧 - 𝑔[j]) / (𝑧 - conj(𝑔[j]))
            ∏12 = Φ[j]
            ∏21 = conj(Φ[j]) * ∏11
            ∏22 = one(CT)
            result *= [∏11 ∏12; ∏21 ∏22]
        end

        return result
    end
    𝒜vec = [_calc_abcd(𝑧) for 𝑧 in 𝑚]
    𝒜 = [𝒜vec[k][i, j] for i in 1:2, j in 1:2, k in 1:nmesh]
    return 𝒜
end

using ACFlowSensitivity
alg = NAC()
T = ComplexF64
grid = T.(1:10)
mesh = T.(1:10)
Φ = rand(T, 10)
𝒜 = calc_abcd(grid, mesh, Φ, alg)

function tf(G)
    res = calc_abcd(grid, mesh, G, alg)
    return norm(res)
end

G = rand(T, 10)
tf(G)
gradient(tf, G)
