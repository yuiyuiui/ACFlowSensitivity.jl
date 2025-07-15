# Consider n-site 1 demension local Ising model with J = 1, H = -\sum \sigma_j \sigma_{j+1}
using SparseArrays, LinearAlgebra, ACFlowSensitivity
using Random, Plots

function example()
    Random.seed!(6)
    nsite = 3
    β = 10.0
    nGFV = 20
    T = Float64
    function hamiltonian(nsite::Int)
        N = 1 << nsite  # 2^nsite
        diag = zeros(Float64, N)
        for state in 0:(N - 1)
            e = 0.0
            for i in 0:(nsite - 2)
                b1 = (state >> i) & 1 == 1 ? 1.0 : -1.0
                b2 = (state >> (i + 1)) & 1 == 1 ? 1.0 : -1.0
                e += -(b1 * b2)
            end
            diag[state + 1] = e
        end
        return spdiagm(0 => diag)
    end

    H = Matrix(hamiltonian(nsite))

    N = 1 << nsite
    A0 = rand(N, N)
    A0 += A0'
    B0 = rand(N, N)
    B0 += B0'
    Z = tr(exp(-β * H))
    A(t) = exp(im * t * H) * A0 * exp(-im * t * H)

    Ave(f::Function, x) = tr(exp(-β * H) * f(x)) / Z

    GAB(τ) = Ave(x -> A(-im * x) * B0, τ)

    SAB(t) = Ave(x -> A(x) * B0, t)

    GFV = Complex{T}[]
    wn = (collect(0:(nGFV - 1)) .+ T(1 // 2)) * 2π / β

    for i in 1:nGFV
        @show i
        Giwn = ACFlowSensitivity.integral(x -> GAB(x) * exp(-im * wn[i] * x), T(0), β)
        push!(GFV, Giwn)
    end

    alg = MaxEntChi2kink(; model_type="Gaussian")
    ctx = CtxData(β, nGFV)

    mesh, reA = solve(GFV, ctx, alg)

    return mesh, reA
end

mesh, reA = example()
plot(mesh, reA)
