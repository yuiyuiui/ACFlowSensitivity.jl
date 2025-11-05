#
# Project : Gardenia
# Source  : nac.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/09/30
#

#
# Note:
#
# The following codes for the NevanAC solver are mostly adapted from
#
#     https://github.com/SpM-lab/Nevanlinna.jl
#
# See
#
#     Nevanlinna.jl: A Julia implementation of Nevanlinna analytic continuation
#     Kosuke Nogaki, Jiani Fei, Emanuel Gull, Hiroshi Shinaoka
#     SciPost Phys. Codebases 19 (2023)
#
# for more details. And we thank Dr. Shuang Liang for her help.
#

#=
### *Customized Structs* : *NevanAC Solver*
=#

"""
    NevanACContext

Mutable struct. It is used within the NevanAC solver only.

### Members
* Gᵥ   -> Input data for correlator.
* grid -> Grid for input data.
* mesh -> Mesh for output spectrum.
* Φ    -> `Φ` vector in Schur algorithm.
* 𝒜    -> Coefficients matrix `abcd` in Schur algorithm.
* ℋ    -> Hardy matrix for Hardy basis optimization.
* 𝑎𝑏   -> Coefficients matrix for expanding `θ` with Hardy basis.
* hmin -> Minimal value of the order of Hardy basis functions.
* hopt -> Optimal value of the order of Hardy basis functions.
"""
mutable struct NevanACContext
    Gᵥ::Vector{APC}
    grid::Vector{APC}
    mesh::Vector{APC}
    Φ::Vector{APC}
    𝒜::Array{APC,3}
    ℋ::Array{APC,2}
    𝑎𝑏::Vector{ComplexF64}
    hmin::Int
    hopt::Int
end

#=
### *Global Drivers*
=#

"""
    solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::NAC) where{T<:Real}

Solve the analytic continuation problem by the NAC method. It is the driver for the NAC solver.

This solver suits Matsubara Green's functions for fermionic systems. It
can not be used directly to treat the bosonic correlators. It will return
A(ω) all the time, similar to the StochPX and BarRat solvers.

This solver is numerically unstable. Sometimes it is hard to get converged
solution, especially when the noise is medium.

### Arguments
* GFV -> A vector of complex numbers, containing the input data for the NAC solver.
* ctx -> A CtxData struct, containing the context data for the NAC solver.
* alg -> A NAC struct, containing the algorithm data for the NAC solver.

### Returns
* mesh -> Real frequency mesh, ω.
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::NAC) where {T<:Real}
    ctx.spt isa Delta && alg.hardy &&
        error("Hardy basis optimization is used for Cont spectrum")
    println("[ NevanAC ]")
    nac = init(GFV, ctx, alg)
    run!(nac, alg)
    Aout, _ = last(nac)
    if ctx.spt isa Cont
        return T.(Aout)
    elseif ctx.spt isa Delta
        idx = find_peaks(ctx.mesh.mesh, Aout, ctx.fp_mp; wind=ctx.fp_ww)
        p = ctx.mesh.mesh[idx]
        γ = pG2γ(p, GFV, ctx.iwn)
        println("poles: ", p)
        println("gamma: ", γ)
        return T.(Aout), (p, γ)
    else
        error("Unsupported spectral function type")
    end
end

"""
    init(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::NAC) where{T<:Real}

Initialize the NevanAC solver and return a NevanACContext struct.

### Arguments
* GFV -> A vector of complex numbers, containing the input data for the NevanAC solver.
* ctx -> A CtxData struct, containing the context data for the NevanAC solver.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.

### Returns
* mec -> A NevanACContext struct.
"""
function init(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::NAC) where {T<:Real}
    # Setup numerical precision. Note that the NAC method is extremely
    # sensitive to the float point precision.
    setprecision(128)

    # Convert the input data to APC, i.e., Complex{BigFloat}.
    ωₙ = APC.(ctx.wn * im)
    Gₙ = APC.(GFV)

    # Evaluate the optimal value for the size of input data.
    # Here we just apply the Pick criterion.
    ngrid = calc_noptim(ωₙ, Gₙ, alg.pick)

    # Prepera input data
    Gᵥ = calc_mobius(-Gₙ[1:ngrid])
    reverse!(Gᵥ)
    println("Postprocess input data: ", length(Gᵥ), " points")

    # Prepare grid for input data
    grid = APC.(ctx.wn)
    resize!(grid, ngrid)
    reverse!(grid)
    println("Build grid for input data: ", length(grid), " points")

    # Prepare mesh for output spectrum
    mesh = APC.(ctx.mesh.mesh)
    println("Build mesh for spectrum: ", length(mesh), " points")

    # Precompute key quantities to accelerate the computation
    Φ, 𝒜, ℋ, 𝑎𝑏 = precompute(grid, mesh, Gᵥ, alg)
    println("Precompute key matrices")

    # Actually, the NevanACContext struct already contains enough
    # information to build the Nevanlinna interpolant and get the
    # spectrum, but Hardy basis optimization is needed to smooth
    # the results further.
    return NevanACContext(Gᵥ, grid, mesh, Φ, 𝒜, ℋ, 𝑎𝑏, 1, 1)
end

"""
    run!(nac::NevanACContext, alg::NAC)

Perform Hardy basis optimization to smooth the spectrum. the members `ℋ`,
`𝑎𝑏`, `hmin`, and `hopt` of the NevanACContext struct (`nac`) should be
updated in this function.

### Arguments
* nac -> A NevanACContext struct.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.

### Returns
N/A
"""
function run!(nac::NevanACContext, alg::NAC)
    hardy = alg.hardy
    #
    if hardy
        println("Activate Hardy basis optimization")

        # Determine the minimal Hardy order (`hmin`), update `ℋ` and `𝑎𝑏`.
        calc_hmin!(nac, alg)

        # Determine the optimal Hardy order (`hopt`), update `ℋ` and `𝑎𝑏`.
        calc_hopt!(nac, alg)
    end
end

"""
    last(nac::NevanACContext)

Postprocess the results generated during the Nevanlinna analytical
continuation simulations.

### Arguments
* nac -> A NevanACContext struct.

### Returns
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function last(nac::NevanACContext)
    # Calculate full response function on real axis and write them
    # Note that _G is actually 𝑁G, so there is a `-` symbol for the
    # return value.
    _G = ComplexF64.(calc_green(nac.𝒜, nac.ℋ, nac.𝑎𝑏))

    # Calculate and write the spectral function
    Aout = Float64.(imag.(_G) ./ π)

    return Aout, -_G
end

"""
    precompute(
        grid::Vector{APC},
        mesh::Vector{APC},
        Gᵥ::Vector{APC},
        alg::NAC
    )

Precompute some key quantities, such as `Φ`, `𝒜`, `ℋ`, and `𝑎𝑏`. Note
that `Φ` and `𝒜` won't be changed any more. But `ℋ` and `𝑎𝑏` should be
updated by the Hardy basis optimization to get a smooth spectrum. Here
`Gᵥ` is input data, `grid` is the grid for input data, and `mesh` is
the mesh for output spectrum.

### Arguments
See above explanations.

### Returns
* Φ -> `Φ` vector in Schur algorithm.
* 𝒜 -> Coefficients matrix `abcd` in Schur algorithm.
* ℋ -> Hardy matrix for Hardy basis optimization.
* 𝑎𝑏 -> Coefficients matrix for expanding `θ` with Hardy basis.

See also: [`NevanACContext`](@ref).
"""
function precompute(grid::Vector{APC},
                    mesh::Vector{APC},
                    Gᵥ::Vector{APC},
                    alg::NAC)
    # Evaluate ϕ and `abcd` matrices
    Φ = calc_phis(grid, Gᵥ)
    𝒜 = calc_abcd(grid, mesh, Φ, alg)

    # Allocate memory for evaluating θ
    # The initial Hardy order is just 1.
    ℋ = calc_hmatrix(mesh, 1, alg)
    𝑎𝑏 = zeros(ComplexF64, 2)

    return Φ, 𝒜, ℋ, 𝑎𝑏
end

"""
    calc_mobius(z::Vector{APC})

A direct Mobius transformation.

### Arguments
* z -> Complex vector.

### Returns
* val -> φ(z), Mobius transformation of z.
"""
function calc_mobius(z::Vector{APC})
    return @. (z - im) / (z + im)
end

"""
    calc_inv_mobius(z::Vector{APC})

An inverse Mobius transformation.

### Arguments
* z -> Complex vector.

### Returns
* val -> φ⁻¹(z), inverse Mobius transformation of z.
"""
function calc_inv_mobius(z::Vector{APC})
    return @. im * (one(APC) + z) / (one(APC) - z)
end

"""
    calc_pick(k::Int, ℎ::Vector{APC}, λ::Vector{APC})

Try to calculate the Pick matrix, anc check whether it is a positive
semidefinite matrix. See Eq. (5) in Fei's NAC paper.

### Arguments
* k -> Size of the Pick matrix.
* ℎ -> Vector ℎ. It is actually 𝑧.
* λ -> Vector λ. It is actually 𝒢(𝑧).

### Returns
* success -> Test that a factorization of the Pick matrix succeeded.
"""
function calc_pick(k::Int, ℎ::Vector{APC}, λ::Vector{APC})
    pick = zeros(APC, k, k)

    # Calculate the Pick matrix
    for j in 1:k
        for i in 1:k
            num = one(APC) - λ[i] * conj(λ[j])
            den = one(APC) - ℎ[i] * conj(ℎ[j])
            pick[i, j] = num / den
        end
        pick[j, j] += APC(1e-250)
    end

    # Cholesky decomposition
    return issuccess(cholesky(pick; check=false))
end

"""
    calc_phis(grid::Vector{APC}, Gᵥ::Vector{APC})

Try to calculate the Φ vector, which is used to calculate the 𝒜 matrix.
Note that Φ should not be changed anymore once it has been established.

### Arguments
* grid -> Grid in imaginary axis for input Green's function.
* Gᵥ -> Input Green's function.

### Returns
* Φ -> `Φ` vector in Schur algorithm.
"""
function calc_phis(grid::Vector{APC}, Gᵥ::Vector{APC})
    ngrid = length(grid)

    # Allocate memory
    Φ = APC[]
    𝑔 = grid * im

    # Initialize the `abcd` matrix
    𝒜 = [i == j ? APC(1) : APC(0) for i in 1:2, j in 1:2, k in 1:ngrid]

    # Evaluate Φ using recursive algorithm
    Φ = vcat(Φ, Gᵥ[1])
    for j in 1:(ngrid - 1)
        for k in (j + 1):ngrid
            ∏11 = (𝑔[k] - 𝑔[j]) / (𝑔[k] - conj(𝑔[j]))
            ∏12 = Φ[j]
            ∏21 = conj(Φ[j]) * ∏11
            ∏22 = one(APC)
            M = [∏11 ∏12; ∏21 ∏22]
            𝒜 = slicerightmul!(𝒜, M, k)
        end
        num = 𝒜[1, 2, j + 1] - 𝒜[2, 2, j + 1] * Gᵥ[j + 1]
        den = 𝒜[2, 1, j + 1] * Gᵥ[j + 1] - 𝒜[1, 1, j + 1]
        Φ = vcat(Φ, num / den)
    end

    return Φ
end

"""
    calc_abcd(grid::Vector{APC}, mesh::Vector{APC}, Φ::Vector{APC}, alg::NAC)

Try to calculate the coefficients matrix abcd (here it is called 𝒜),
which is then used to calculate θ. See Eq. (8) in Fei's NAC paper.

### Arguments
* grid -> Grid in imaginary axis for input Green's function.
* mesh -> Real frequency mesh.
* Φ -> Φ vector calculated by `calc_phis()`.

### Returns
* 𝒜 -> Coefficients matrix `abcd` in Schur algorithm.
"""
function calc_abcd(grid::Vector{APC}, mesh::Vector{APC}, Φ::Vector{APC}, alg::NAC)
    eta = APC(alg.eta)
    ngrid = length(grid)
    nmesh = length(mesh)
    𝑔 = grid * im
    𝑚 = mesh .+ im * eta
    𝒜 = zeros(APC, 2, 2, nmesh)

    function _calc_abcd(𝑧)
        result = Matrix{APC}(I, 2, 2)
        for j in 1:ngrid
            ∏11 = (𝑧 - 𝑔[j]) / (𝑧 - conj(𝑔[j]))
            ∏12 = Φ[j]
            ∏21 = conj(Φ[j]) * ∏11
            ∏22 = one(APC)
            result *= [∏11 ∏12; ∏21 ∏22]
        end

        return result
    end
    𝒜vec = [_calc_abcd(𝑧) for 𝑧 in 𝑚]
    𝒜 = [𝒜vec[k][i, j] for i in 1:2, j in 1:2, k in 1:nmesh]
    return 𝒜
end

"""
    calc_hbasis(z::APC, k::Int)

Try to calculate the Hardy basis ``f^k(z)``.

### Arguments
* z -> A complex variable.
* k -> Current order for the Hardy basis.

### Returns
See above explanations.
"""
function calc_hbasis(z::APC, k::Int)
    w = (z - im) / (z + im)
    return 1 / (sqrt(π) * (z + im)) * w^k
end

"""
    calc_hmatrix(mesh::Vector{APC}, H::Int, alg::NAC)

Try to calculate ``[f^k(z), f^k(z)^*]`` for 0 ≤ 𝑘 ≤ 𝐻-1, which is
called the hardy matrix (ℋ) and is used to evaluate θₘ₊₁.

### Arguments
* mesh -> Real frequency mesh.
* H -> Maximum order for the Hardy basis.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.
### Returns
* ℋ -> Hardy matrix for Hardy basis optimization.
"""
function calc_hmatrix(mesh::Vector{APC}, H::Int, alg::NAC)
    # Build real axis
    eta = APC(alg.eta)
    𝑚 = mesh .+ eta * im

    # Allocate memory for the Hardy matrix
    nmesh = length(mesh)
    ℋ = zeros(APC, nmesh, 2 * H)

    # Build the Hardy matrix
    for k in 1:H
        ℋ[:, 2 * k - 1] .= calc_hbasis.(𝑚, k - 1)
        ℋ[:, 2 * k] .= conj(ℋ[:, 2 * k - 1])
    end

    return ℋ
end

"""
    calc_theta(𝒜::Array{APC,3}, ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64})

Try to calculate the contractive function θ(z). 𝒜 is the coefficients
matrix abcd, ℋ is the Hardy matrix, and 𝑎𝑏 are complex coefficients
for expanding θₘ₊₁. See Eq. (7) in Fei's NAC paper.

### Arguments
* 𝒜  -> Matrix 𝑎𝑏𝑐𝑑.
* ℋ  -> Hardy matrix.
* 𝑎𝑏 -> Expansion coefficients 𝑎 and 𝑏 for the contractive function θ.

### Returns
See above explanations.
"""
function calc_theta(𝒜::Array{APC,3}, ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64})
    # Well, we should calculate θₘ₊₁ at first.
    θₘ₊₁ = ℋ * 𝑎𝑏

    # Then we evaluate θ according Eq. (7)
    num = 𝒜[1, 1, :] .* θₘ₊₁ .+ 𝒜[1, 2, :]
    den = 𝒜[2, 1, :] .* θₘ₊₁ .+ 𝒜[2, 2, :]
    θ = num ./ den

    return θ
end

"""
    calc_green(𝒜::Array{APC,3}, ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64})

Firstly we try to calculate θ. Then θ is back transformed to a Nevanlinna
interpolant via the inverse Mobius transform. Here, `𝒜` (`abcd` matrix),
`ℋ` (Hardy matrix), and `𝑎𝑏` are used to evaluate θ.

### Arguments
* 𝒜  -> Matrix 𝑎𝑏𝑐𝑑.
* ℋ  -> Hardy matrix.
* 𝑎𝑏 -> Expansion coefficients 𝑎 and 𝑏 for the contractive function θ.

### Returns
Gout -> Retarded Green's function, G(ω).
"""
function calc_green(𝒜::Array{APC,3}, ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64})
    θ = calc_theta(𝒜, ℋ, 𝑎𝑏)
    gout = calc_inv_mobius(θ)

    return gout
end

"""
    calc_noptim(ωₙ::Vector{APC}, Gₙ::Vector{APC}, pick::Bool)

Evaluate the optimal value for the size of input data (how may frequency
points are actually used in the analytic continuation simulations) via
the Pick criterion.

### Arguments
* ωₙ -> Matsubara frequency points (the 𝑖 unit is not included).
* Gₙ -> Matsubara Green's function.
* pick -> Whether to use the Pick criterion.

### Returns
* ngrid -> Optimal number for the size of input data.
"""
function calc_noptim(ωₙ::Vector{APC}, Gₙ::Vector{APC}, pick::Bool)
    # Get size of input data
    ngrid = length(ωₙ)

    # Check whether the Pick criterion is applied
    if !pick
        return ngrid
    end

    # Apply invertible Mobius transformation. We actually work at
    # the \bar{𝒟} space.
    𝓏 = calc_mobius(ωₙ)
    𝒢 = calc_mobius(-Gₙ)

    # Find the optimal value of k until the Pick criterion is violated
    k = 0
    success = true
    while success && k ≤ ngrid
        k += 1
        success = calc_pick(k, 𝓏, 𝒢)
    end

    # Return the optimal value for the size of input data
    if !success
        println("The size of input data is optimized to $(k-1)")
        return k - 1
    else
        println("The size of input data is optimized to $(ngrid)")
        return ngrid
    end
end

"""
    calc_hmin!(nac::NevanACContext, alg::NAC)

Try to perform Hardy basis optimization. Such that the Hardy matrix ℋ
and the corresponding coefficients 𝑎𝑏 are updated. They are used to
calculate θ, which is then back transformed to generate smooth G (i.e.,
the spectrum) at real axis.

This function will determine the minimal value of H (hmin). Of course,
ℋ and 𝑎𝑏 in NevanACContext struct are also changed.

### Arguments
* nac -> A NevanACContext struct.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.

### Returns
N/A
"""
function calc_hmin!(nac::NevanACContext, alg::NAC)
    hmax = alg.hmax

    h = 1
    while h ≤ hmax
        println("H (Order of Hardy basis) -> $h")

        # Prepare initial ℋ and 𝑎𝑏
        ℋ = calc_hmatrix(nac.mesh, h, alg)
        𝑎𝑏 = zeros(ComplexF64, 2 * h)

        # Hardy basis optimization
        causality, optim = hardy_optimize!(nac, ℋ, 𝑎𝑏, h, alg)

        # Check whether the causality is preserved and the
        # optimization is successful.
        if causality && optim
            nac.hmin = h
            break
        else
            h = h + 1
        end
    end
end

"""
    calc_hopt!(nac::NevanACContext, alg::NAC)

Try to perform Hardy basis optimization. Such that the Hardy matrix ℋ
and the corresponding coefficients 𝑎𝑏 are updated. They are used to
calculate θ, which is then back transformed to generate smooth G (i.e.,
the spectrum) at real axis.

This function will determine the optimal value of H (hopt). Of course,
ℋ and 𝑎𝑏 in NevanACContext struct are also changed. Here, H means order
of the Hardy basis.

### Arguments
* nac -> A NevanACContext struct.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.

### Returns
N/A
"""
function calc_hopt!(nac::NevanACContext, alg::NAC)
    hmax = alg.hmax

    for h in (nac.hmin + 1):hmax
        println("H (Order of Hardy basis) -> $h")

        # Prepare initial ℋ and 𝑎𝑏
        ℋ = calc_hmatrix(nac.mesh, h, alg)
        𝑎𝑏 = copy(nac.𝑎𝑏)
        push!(𝑎𝑏, zero(ComplexF64))
        push!(𝑎𝑏, zero(ComplexF64))
        @assert size(ℋ)[2] == length(𝑎𝑏)

        # Hardy basis optimization
        causality, optim = hardy_optimize!(nac, ℋ, 𝑎𝑏, h, alg)

        # Check whether the causality is preserved and the
        # optimization is successful.
        if !(causality && optim)
            break
        end
    end
end

"""
    hardy_optimize!(
        nac::NevanACContext,
        ℋ::Array{APC,2},
        𝑎𝑏::Vector{ComplexF64},
        H::Int,
        alg::NAC
    )

For given Hardy matrix ℋ, try to update the expanding coefficients 𝑎𝑏
by minimizing the smooth norm.

### Arguments
* nac -> A NevanACContext struct.
* ℋ   -> Hardy matrix, which contains the Hardy basis.
* 𝑎𝑏  -> Expansion coefficients 𝑎 and 𝑏 for the contractive function θ.
* H   -> Current order of the Hardy basis.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.

### Returns
* causality -> Test whether the solution is causal.
* converged -> Check whether the optimization is converged.
"""
function hardy_optimize!(nac::NevanACContext,
                         ℋ::Array{APC,2},
                         𝑎𝑏::Vector{ComplexF64},
                         H::Int,
                         alg::NAC)
    # Function call to the smooth norm.
    function 𝑓(x::Vector{ComplexF64})
        return smooth_norm(nac, ℋ, x, alg)
    end

    # Function call to the gradient of the smooth norm.
    #
    # Here we adopt the Zygote package, which implements an automatic
    # differentiation algorithm, to evaluate the gradient of the smooth
    # norm. Of course, we can turn to the finite difference algorithm,
    # which is less efficient.
    function 𝐽!(J::Vector{ComplexF64}, x::Vector{ComplexF64})
        #J .= Zygote.gradient(𝑓, x)[1]

        # Finite difference algorithm
        return J .= fdgradient(𝑓, x)
    end

    # Perform numerical optimization by the BFGS algorithm.
    # If it is failed, please turn to the Optim.jl package.
    # A simplified version is implemented in math.jl.
    res = bfgs(𝑓, 𝐽!, 𝑎𝑏; max_iter=500)

    # Check whether the BFGS algorithm is converged
    if !converged(res)
        @info("Sorry, faild to optimize the smooth norm!")
    end

    # Check causality of the solution
    causality = check_causality(ℋ, res.minimizer)

    # Update ℋ and the corresponding 𝑎𝑏
    if causality && (converged(res))
        nac.hopt = H
        nac.𝑎𝑏 = res.minimizer
        nac.ℋ = ℋ
    end

    return causality, converged(res)
end

"""
    smooth_norm(nac::NevanACContext, ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64}, alg::NAC)

Establish the smooth norm, which is used to improve the smoothness of
the output spectrum. See Fei's paper for more details.

### Arguments
* nac -> A NevanACContext struct.
* ℋ   -> Hardy matrix, which contains the Hardy basis.
* 𝑎𝑏  -> Expansion coefficients 𝑎 and 𝑏 for the contractive function θ.
* alg -> A NevanAC struct, containing the algorithm data for the NevanAC solver.
### Returns
* 𝐹 -> Value of smooth norm.
"""
function smooth_norm(nac::NevanACContext, ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64}, alg::NAC)
    # Get regulation parameter
    α = APC(alg.alpha)

    # Generate output spectrum
    _G = calc_green(nac.𝒜, ℋ, 𝑎𝑏)
    A = Float64.(imag.(_G) ./ π)

    # Normalization term
    𝑓₁ = trapz(nac.mesh, A)

    # Smoothness term
    sd = second_derivative(nac.mesh, A)
    x_sd = nac.mesh[2:(end - 1)]
    𝑓₂ = trapz(x_sd, abs.(sd) .^ 2)

    # Assemble the final smooth norm
    𝐹 = abs(1 - 𝑓₁)^2 + α * 𝑓₂

    return Float64(𝐹)
end

"""
    check_pick(wn::Vector{APC}, gw::Vector{APC}, Nopt::Int)

Check whether the input data are valid (the Pick criterion is satisfied).
Here, `wn` is the Matsubara frequency, `gw` is the Matsubara function,
and `Nopt` is the optimized number of Matsubara data points.

### Arguments
See above explanations.

### Returns
N/A
"""
function check_pick(wn::Vector{APC}, gw::Vector{APC}, Nopt::Int)
    freq = calc_mobius(wn[1:Nopt])
    val = calc_mobius(-gw[1:Nopt])

    success = calc_pick(Nopt, val, freq)
    #
    if success
        println("Pick matrix is positive semi-definite.")
    else
        println("Pick matrix is non positive semi-definite matrix in Schur method.")
    end
end

"""
    check_causality(ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64})

Check causality of the Hardy coefficients `𝑎𝑏`.

### Arguments
* ℋ -> Hardy matrix for Hardy basis optimization.
* 𝑎𝑏 -> Coefficients matrix for expanding `θ` with Hardy basis.

### Returns
* causality -> Test whether the Hardy coefficients are causal.
"""
function check_causality(ℋ::Array{APC,2}, 𝑎𝑏::Vector{ComplexF64})
    θₘ₊₁ = ℋ * 𝑎𝑏

    max_theta = findmax(abs.(θₘ₊₁))[1]

    if max_theta <= 1.0
        println("max_theta = ", max_theta)
        println("Hardy optimization was success.")
        causality = true
    else
        println("max_theta = ", max_theta)
        println("Hardy optimization was failure.")
        causality = false
    end

    return causality
end

#---------------------------------
# solve differentiation
function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::NAC;
                   diffonly::Bool=false) where {T<:Real}
    if ctx.spt isa Cont
        return Adiff(GFV, ctx, alg; diffonly=diffonly)
    elseif ctx.spt isa Delta
        return pγdiff(GFV, ctx, alg)
    else
        error("Unsupported spectral function type")
    end
end
