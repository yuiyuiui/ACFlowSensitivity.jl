#
# Note:
#
# The following codes for the Prony approximation are mostly adapted from
#
#     https://github.com/Green-Phys/PronyAC
#
# See
#
#     Minimal Pole Representation and Controlled Analytic Continuation
#     of Matsubara Response Functions
#     Lei Zhang and Emanuel Gull
#     arXiv:2312.10576 (2024)
#
# for more details.
#

#=
### *Customized Structs* : *PronyApproximation*
=#

#=
*Remarks* :

**Prony interpolation**

Our input data consists of an odd number ``2N + 1`` of Matsubara points
``G(i\omega_n)`` that are uniformly spaced. Prony's interpolation method
interpolates ``G_k`` as a sum of exponentials

```math
G_k = \sum^{N-1}_{i=0} w_i \gamma^k_i,
```

where ``0 \le k \le 2N``, ``w_i`` denote complex weights and ``\gamma_i``
corresponding nodes.

---

**Prony approximation**

Prony's interpolation method is unstable. We therefore employs a Prony
approximation, rather than an interpolation of ``G``. For the physical
Matsubara functions, which decay in magnitude to zero for
``i\omega_n \to i\infty``, only ``K \propto \log{1/\varepsilon}`` out of
all ``N`` nodes in the Prony approximation have weights
``|w_i| > \varepsilon``. Thus, we have

```math
\left|G_k - \sum^{K-1}_{i=0} w_i \gamma^k_i\right| \le \varepsilon,
```

for all ``0 \le k \le 2N``.
=#

"""
    PronyApproximation

Mutable struct. Prony approximation to a complex-valued Matsubara function.

### Members
* 𝑁ₚ -> Number of nodes for Prony approximation.
* ωₚ -> Non-negative Matsubara frequency.
* 𝐺ₚ -> Complex values at ωₚ.
* Γₚ -> Nodes for Prony approximation, ``γ_i``.
* Ωₚ -> Weights for Prony approximation, ``w_i``.
"""
mutable struct PronyApproximation{T<:Real,S<:Int} <: Function
    𝑁ₚ::S
    ωₚ::Vector{T}
    𝐺ₚ::Vector{Complex{T}}
    Γₚ::Vector{Complex{T}}
    Ωₚ::Vector{Complex{T}}
end

"""
    (𝑝::PronyApproximation)(w::Vector{T}) where {T}

Evaluate the Prony approximation at `w`.

### Arguments
* w -> w ∈ ℝ.

### Returns
* val -> 𝑝.(w).
"""
function (𝑝::PronyApproximation)(w::Vector{T}) where {T}
    x₀ = @. (w - w[1]) / (w[end] - w[1])
    𝔸 = zeros(Complex{T}, length(x₀), length(𝑝.Ωₚ))
    #
    for i in eachindex(x₀)
        @. 𝔸[i, :] = 𝑝.Γₚ ^ (2 * 𝑝.𝑁ₚ * x₀[i])
    end
    #
    return 𝔸 * 𝑝.Ωₚ
end

"""
    PronyApproximation(
        𝑁ₚ :: Int,
        ωₚ :: Vector{T},
        𝐺ₚ :: Vector{Complex{T}},
        v  :: Vector{Complex{T}}
    )

Construct a `PronyApproximation` type interpolant function. Once it is
available, then it can be used to produce a smooth G at given ω.

This function should not be called directly by the users.

### Arguments
* 𝑁ₚ -> Number of nodes for Prony approximation.
* ωₚ -> Non-negative Matsubara frequency (postprocessed).
* 𝐺ₚ -> Complex values at ωₚ (postprocessed).
* v  -> Selected vector from the orthogonal matrix `V`.

### Returns
* pa -> A PronyApproximation struct.
"""
function PronyApproximation(𝑁ₚ::Int, ωₚ::Vector{T}, 𝐺ₚ::Vector{Complex{T}},
                            v::Vector{Complex{T}}) where {T}
    # Evaluate cutoff for Γₚ
    Λ = 1+T(1//2) / 𝑁ₚ

    # Evaluate Γₚ and Ωₚ
    Γₚ = prony_gamma(v, Λ)
    Ωₚ = prony_omega(𝐺ₚ, Γₚ)

    # Sort Γₚ and Ωₚ
    idx_sort = sortperm(abs.(Ωₚ))
    reverse!(idx_sort)
    Ωₚ = Ωₚ[idx_sort]
    Γₚ = Γₚ[idx_sort]

    # Return a PronyApproximation struct
    return PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, Γₚ, Ωₚ)
end

"""
    PronyApproximation(ω₁::Vector{T}, 𝐺₁::Vector{Complex{T}}, ε) where {T}

Construct a `PronyApproximation` type interpolant function. Once it is
available, then it can be used to produce a smooth G at ω.

If the noise level of the input data is known, this function is a good
choice. The parameter `ε` can be set to the noise level.

### Arguments
* ω₁ -> Non-negative Matsubara frequency (raw values).
* 𝐺₁ -> Complex values at ωₚ (raw values).
* ε  -> Threshold for the Prony approximation.

### Returns
* pa -> A PronyApproximation struct.
"""
function PronyApproximation(ω₁::Vector{T}, 𝐺₁::Vector{Complex{T}}, ε) where {T}
    # Preprocess the input data to get the number of nodes, frequency
    # points ωₚ, and Matsubara data 𝐺ₚ.
    𝑁ₚ, ωₚ, 𝐺ₚ = prony_data(ω₁, 𝐺₁)

    # Perform singular value decomposition and select reasonable `v`.
    S, V = prony_svd(𝑁ₚ, 𝐺ₚ)
    v = prony_v(V, prony_idx(S, ε))

    return PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, v)
end

"""
    PronyApproximation(ω₁::Vector{T}, 𝐺₁::Vector{Complex{T}}) where {T}

Construct a `PronyApproximation` type interpolant function. Once it is
available, then it can be used to produce a smooth G at ω. Note that this
function employs a smart and iterative algorithm to determine the optimal
Prony approximation.

This function is time-consuming. But if the noise level of the input data
is unknown, this function is useful.

### Arguments
* ω₁ -> Non-negative Matsubara frequency (raw values).
* 𝐺₁ -> Complex values at ωₚ (raw values).

### Returns
* pa -> A PronyApproximation struct.
"""
function PronyApproximation(ω₁::Vector{T}, 𝐺₁::Vector{Complex{T}}) where {T}
    # Preprocess the input data to get the number of nodes, frequency
    # points ωₚ, and Matsubara data 𝐺ₚ.
    𝑁ₚ, ωₚ, 𝐺ₚ = prony_data(ω₁, 𝐺₁)

    # Perform singular value decomposition
    S, V = prony_svd(𝑁ₚ, 𝐺ₚ)

    # Next we should determine the optimal `v`
    #
    # (1) Find maximum index for the exponentially decaying region.
    exp_idx = prony_idx(S)
    #
    # (2) Find minimum index
    ε = 1000 * S[exp_idx]
    new_idx = findfirst(x -> x < ε, S)
    #
    # (3) Create lists for chosen indices and the corresponding errors.
    idxrange = range(new_idx, min(exp_idx + 10, length(S)))
    idx_list = collect(idxrange)
    err_list = zeros(T, length(idx_list))
    #
    # (4) Create a list of pseudo-PronyApproximations, and then evaluate
    # their reliabilities and accuracies.
    for i in eachindex(idx_list)
        idx = idx_list[i]
        #
        # Extract `v`
        v = prony_v(V, idx)
        #
        # Reproduce G using the pseudo PronyApproximation
        𝐺ₙ = PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, v)(ωₚ)
        #
        # Evaluate the difference and record it
        err_ave = mean(abs.(𝐺ₙ - 𝐺ₚ))
        err_list[i] = err_ave
    end
    #
    # (5) Find the optimal `v`, which should minimize |𝐺ₙ - 𝐺ₚ|
    idx = idx_list[argmin(err_list)]
    v = prony_v(V, idx)
    println("The optimal Prony approximation is $idx")

    return PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, v)
end

"""
    prony_data(ω₁::Vector{T}, 𝐺₁::Vector{Complex{T}}) where {T}

Prepare essential data for the later Prony approximation. It will return
the number of nodes 𝑁ₚ, frequency mesh ωₚ, and Green's function data 𝐺ₚ
at this mesh.

### Arguments
* ω₁ -> Non-negative Matsubara frequency (raw values).
* 𝐺₁ -> Complex values at ωₚ (raw values), 𝐺₁ = G(ω₁).

### Returns
See above explanations.
"""
function prony_data(ω₁::Vector{T}, 𝐺₁::Vector{Complex{T}}) where {T}
    # We have to make sure the number of data points is odd.
    osize = length(ω₁)
    nsize = iseven(osize) ? osize - 1 : osize
    #
    𝑁ₚ = div(nsize, 2) # Number of nodes for Prony approximation
    ωₚ = ω₁[1:nsize]   # Matsubara frequency, ωₙ
    𝐺ₚ = 𝐺₁[1:nsize]   # Matsubara Green's function, G(iωₙ)
    #
    return 𝑁ₚ, ωₚ, 𝐺ₚ
end

"""
    prony_svd(𝑁ₚ::Int, 𝐺ₚ::Vector{T}) where {T}

Perform singular value decomposition for the matrix ℋ that is constructed
from 𝐺ₚ. It will return the singular values `S` and orthogonal matrix `V`.

### Arguments
* 𝑁ₚ -> Number of nodes.
* 𝐺ₚ -> Truncated Green's function data.

### Returns
See above explanations.

See also: [`prony_data`](@ref).
"""
function prony_svd(𝑁ₚ::Int, 𝐺ₚ::Vector{T}) where {T}
    ℋ = zeros(T, 𝑁ₚ + 1, 𝑁ₚ + 1)
    #
    for i in 1:(𝑁ₚ + 1)
        ℋ[i, :] = 𝐺ₚ[i:(i + 𝑁ₚ)]
    end
    #
    _, S, V = LinearAlgebra.svd(ℋ)
    println("Singular values: $S")

    return S, V
end

"""
    prony_idx(S::Vector{T}, ε::T) where {T}

The diagonal matrix (singular values) `S` is used to test whether the
threshold `ε` is reasonable and figure out the index for extracting `v`
from `V`.

### Arguments
* S -> Singular values of ℋ.
* ε -> Threshold provided by the users.

### Returns
* idx -> Index for S[idx] < ε.

See also: [`prony_v`](@ref) and [`prony_svd`](@ref).
"""
function prony_idx(S::Vector{T}, ε) where {T}
    # Determine idx, such that S[idx] < ε.
    idx = findfirst(x -> x < ε, S)

    # Check idx
    if isnothing(idx)
        error("Please increase ε and try again! ε ∈ [$(S[1]),$(S[end])]")
    end

    return idx
end

"""
    prony_idx(S::Vector{T}) where {T}

The diagonal matrix (singular values) `S` is used to figure out the index
for extracting `v` from `V`. This function is try to evaluate the maximum
index for the exponentially decaying region of `S`.

### Arguments
* S -> Singular values of ℋ.

### Returns
* idx -> Index for extracting `v` from `V`.

See also: [`prony_v`](@ref).
"""
function prony_idx(S::Vector{T}) where {T}
    n_max = min(3 * floor(Int, log(1.0e12)), floor(Int, 0.8 * length(S)))
    #
    idx_fit = collect(range(ceil(Int, 0.8*n_max), n_max))
    val_fit = S[idx_fit]
    𝔸 = hcat(idx_fit, ones(Int, length(idx_fit)))
    #
    𝑎, 𝑏 = pinv(𝔸) * log.(val_fit)
    𝕊 = exp.(𝑎 .* collect(range(1, n_max)) .+ 𝑏)
    #
    idx = count(S[1:n_max] .> 5 * 𝕊) + 1

    return idx
end

"""
    prony_v(V::Adjoint{T, Matrix{T}}, idx::Int) where {T}

Extract suitable vector `v` from orthogonal matrix `V` according to the
threshold `ε`.

### Arguments
* V -> Orthogonal matrix from singular value decomposition of ℋ.
* idx -> Index for extracting `v` from `V`.

### Returns
* v -> Vector `v` extracted from `V` according to `idx`.

See also: [`prony_svd`](@ref).
"""
function prony_v(V::Adjoint{T,Matrix{T}}, idx::Int) where {T}
    # Extract v from V
    println("Selected vector from orthogonal matrix V: ", idx)
    v = V[:, idx]

    return reverse!(v)
end

"""
    prony_gamma(v::Vector{Complex{T}}, Λ::T) where {T}

Try to calculate Γₚ. Actually, Γₚ are eigenvalues of a matrix constructed
by `v`. `Λ` is a cutoff for Γₚ. Only those Γₚ that are smaller than `Λ`
are kept.

### Arguments
* v -> A vector extracted from `V`.
* Λ -> A cutoff for Γₚ.

### Returns
* Γₚ -> Roots of a polynominal with coefficients given in `v`.

See also: [`prony_v`](@ref).
"""
function prony_gamma(v::Vector{Complex{T}}, Λ::T) where {T}
    # The following codes actually calculate the roots of a polynominal
    # with coefficients given in v. The roots are Γₚ.
    non_zero = findall(!iszero, v)
    trailing_zeros = length(v) - non_zero[end]
    #
    vnew = v[non_zero[1]:non_zero[end]]
    N = length(vnew)
    #
    if N > 1
        A = diagm(-1=>ones(Complex{T}, N - 2))
        @. A[1, :] = -vnew[2:end] / vnew[1]
        roots = eigvals(A)
    else
        roots = []
    end
    #
    Γₚ = vcat(roots, zeros(Complex{T}, trailing_zeros))

    filter!(x -> abs(x) < Λ, Γₚ)

    return Γₚ
end

"""
    prony_omega(𝐺ₚ::Vector{T}, Γₚ::Vector{T}) where {T}

Try to calculate Ωₚ.

### Arguments
* 𝐺ₚ -> Complex values at ωₚ.
* Γₚ -> Nodes for Prony approximation, ``γ_i``.

### Returns
* Ωₚ -> Weights for Prony approximation, ``w_i``.
"""
function prony_omega(𝐺ₚ::Vector{T}, Γₚ::Vector{T}) where {T}
    A = zeros(T, length(𝐺ₚ), length(Γₚ))
    #
    for i in eachindex(𝐺ₚ)
        A[i, :] = Γₚ .^ (i - 1)
    end
    #
    return pinv(A) * 𝐺ₚ
end
