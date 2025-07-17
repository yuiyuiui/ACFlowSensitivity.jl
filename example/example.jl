# Consider n-site 1 demension local Ising model with J = 1, H = -\sum \sigma_j \sigma_{j+1}
using LinearAlgebra, ACFlowSensitivity
using Random, Plots

function example()
    Random.seed!(6)
    nsite = 3
    β = 10.0
    nGFV = 20
    T = Float64

    # Create annihilation operator at site i with Jordan–Wigner string
    function c_op(n::Int, i::Int)
        # Initialize identity operators of correct type
        I2 = Matrix{Float64}(I, 2, 2)
        ops = [I2 for _ in 1:n]
        # Pauli Z and local annihilation operator as Float64 matrices
        σz = Float64[1 0; 0 -1]
        c_local = Float64[0 1; 0 0]
        # Apply parity string
        for j in 1:(i - 1)
            ops[j] = σz
        end
        # Local annihilation at site i
        ops[i] = c_local
        # Kronecker product over all sites
        M = ops[1]
        for k in 2:n
            M = kron(M, ops[k])
        end
        return M
    end

    # Build the dense tight‑binding Hamiltonian
    function fermion_chain(n::Int, t::Float64=1.0)
        dim = 2^n
        H = zeros(Float64, dim, dim)
        for i in 1:(n - 1)
            ci = c_op(n, i)
            cip1 = c_op(n, i + 1)
            H .-= t * (ci' * cip1 + cip1' * ci)
        end
        return H
    end

    H = fermion_chain(nsite)

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
    ctx = CtxData(Cont(), β, nGFV)

    mesh, reA = solve(GFV, ctx, alg)

    return mesh, reA
end

mesh, reA = example()
plot(mesh, reA; label="reconstructed A(w)", title="MaxEntChi2kink", xlabel="w",
     ylabel="A(w)")
