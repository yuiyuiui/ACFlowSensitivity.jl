#
# Project : Gardenia
# Source  : spx.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/09/30
#

#=
### *Customized Structs* : *StochPX Solver*
=#

"""
    StochPXMC

Mutable struct. It is used within the StochPX solver. It includes random
number generator and some counters.

### Members
* rng  -> Random number generator.
* Sacc -> Counter for position-updated (type 1) operation (accepted).
* Stry -> Counter for position-updated (type 1) operation (tried).
* Pacc -> Counter for position-updated (type 2) operation (accepted).
* Ptry -> Counter for position-updated (type 2) operation (tried).
* Aacc -> Counter for amplitude-updated operation (accepted).
* Atry -> Counter for amplitude-updated operation (tried).
* Xacc -> Counter for exchange operation (accepted).
* Xtry -> Counter for exchange operation (tried).

See also: [`StochPXSolver`](@ref).
"""
mutable struct StochPXMC{I<:Int}
    rng::AbstractRNG
    Sacc::I
    Stry::I
    Pacc::I
    Ptry::I
    Aacc::I
    Atry::I
    Xacc::I
    Xtry::I
end

"""
    StochPXElement

Mutable struct. It is used to record the field configurations, which will
be sampled by Monte Carlo sweeping procedure.

For the off-diagonal elements of the matrix-valued Green's function, the
signs of the poles (𝕊) could be negative (-1.0). However, for the other
cases, 𝕊 is always positive (+1.0).

### Members
* P  -> It means the positions of the poles.
* A  -> It means the weights / amplitudes of the poles.
* 𝕊  -> It means the signs of the poles.
"""
mutable struct StochPXElement{I<:Int,T<:Real}
    P::Vector{I}
    A::Vector{T}
    𝕊::Vector{T}
end

"""
    StochPXContext

Mutable struct. It is used within the StochPX solver only.

Note that χ² denotes the goodness-of-fit functional, and Gᵧ denotes the
reproduced correlator. They should be always compatible with P, A, and 𝕊
in the StochPXElement struct.

### Members
* Gᵥ    -> Input data for correlator.
* Gᵧ    -> Generated correlator.
* σ¹    -> Actually 1.0 / σ¹.
* allow -> Allowable indices.
* wn  -> Grid for input data.
* mesh  -> Mesh for output spectrum.
* fmesh -> Very dense mesh for the poles.
* Λ     -> Precomputed kernel matrix.
* Θ     -> Artificial inverse temperature.
* χ²    -> Goodness-of-fit functional for the current configuration.
* χ²ᵥ   -> Vector of goodness-of-fit functional.
* Pᵥ    -> Vector of poles' positions.
* Aᵥ    -> Vector of poles' amplitudes.
* 𝕊ᵥ    -> Vector of poles' signs.
"""
mutable struct StochPXContext{I<:Int,T<:Real}
    Gᵥ::Vector{T}
    Gᵧ::Vector{T}
    σ¹::T
    allow::Vector{I}
    wn::Vector{T}
    mesh::Mesh{T}
    fmesh::Vector{T}
    Λ::Array{T,2}
    Θ::T
    χ²::T
    χ²ᵥ::Vector{T}
    Pᵥ::Vector{Vector{I}}
    Aᵥ::Vector{Vector{T}}
    𝕊ᵥ::Vector{Vector{T}}
end

#=
### *Global Drivers*
=#

"""
    solve(S::StochPXSolver, rd::RawData)

Solve the analytic continuation problem by the stochastic pole expansion.
It is the driver for the StochPX solver. Note that this solver is still
`experimental`. It is useful for analytic continuation of Matsubara data.
In other words, this solver should not be used to handle the imagnary-time
Green's function.

Similar to the BarRat and NevanAC solvers, this solver always returns A(ω).

### Arguments
* S -> A StochPXSolver struct.
* rd -> A RawData struct, containing raw data for input correlator.

### Returns
* mesh -> Real frequency mesh, ω.
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function solve(GFV::Vector{Complex{T}}, ctx::CtxData, alg::SPX) where {T<:Real}
    println("[ StochPX ]")
    ctx.spt isa Cont && (alg.method == "best") &&
        error("SPX with method = \"mean\" is recommended for cont type spectrum")
    ctx.spt isa Delta && (alg.method == "mean") &&
        error("SPX with method = \"best\" is recommended for delta type spectrum")
    mesh = ctx.mesh.mesh
    fine_mesh = collect(range(mesh[1], mesh[end], alg.nfine)) # spx needs high-precise linear grid
    MC = init_mc(alg)
    println("Create infrastructure for Monte Carlo sampling")

    # Initialize Monte Carlo configurations
    SE = init_element(alg, MC.rng, T)
    println("Randomize Monte Carlo configurations")

    # Prepare some key variables
    Gᵥ = vcat(real(GFV), (imag(GFV)))
    Gᵧ, Λ, Θ, χ², χ²ᵥ, Pᵥ, Aᵥ, 𝕊ᵥ = init_context(alg, SE, ctx.wn, fine_mesh, Gᵥ)
    SC = StochPXContext(Gᵥ, Gᵧ, T(1/ctx.σ), collect(1:alg.nfine), ctx.wn, ctx.mesh,
                        fine_mesh,
                        Λ, Θ, χ², χ²ᵥ, Pᵥ, Aᵥ, 𝕊ᵥ)
    println("Initialize context for the StochPX solver")

    return run!(MC, SE, SC, alg)
end

"""
    run!(MC::StochPXMC, SE::StochPXElement, SC::StochPXContext, alg::SPX)

Perform stochastic pole expansion simulation, sequential version.

### Arguments
* MC -> A StochPXMC struct.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
* Gᵣ -> Reproduced Green's function, G(iωₙ).
"""
function run!(MC::StochPXMC{I}, SE::StochPXElement{I,T}, SC::StochPXContext{I,T},
              alg::SPX) where {I<:Int,T<:Real}

    # Setup essential parameters
    ntry = alg.ntry
    nstep = alg.nstep

    # Warmup the Monte Carlo engine
    println("Start thermalization...")
    for _ in 1:nstep
        sample!(1, MC, SE, SC, alg)
    end

    # Sample and collect data
    println("Start stochastic sampling...")
    for t in 1:ntry
        # Reset Monte Carlo counters
        reset_mc!(MC)

        # Reset Monte Carlo field configuration
        reset_element!(MC.rng, SE, alg)

        # Reset Gᵧ and χ² in SC (StochPXContext)
        reset_context!(t, SE, SC, alg)

        # Apply simulated annealing algorithm
        for _ in 1:nstep
            sample!(t, MC, SE, SC, alg)
        end

        # Show the best χ² (the smallest) for the current attempt
        t%10 == 0 && println("try = $t -> [χ² = $(SC.χ²ᵥ[t])]")
    end

    # Generate spectral density from Monte Carlo field configuration
    return average!(SC, alg)
end

"""
    average!(SC::StochPXContext, alg::SPX)

Postprocess the results generated during the stochastic pole expansion
simulations. It will calculate the spectral functions, real frequency
Green's function, and imaginary frequency Green's function.

### Arguments
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
* Gᵣ -> Reproduced Green's function, G(iωₙ).
"""
function average!(SC::StochPXContext{I,T}, alg::SPX) where {I<:Int,T<:Real}
    # Setup essential parameters
    nmesh = length(SC.mesh.mesh)
    method = alg.method
    ntry = alg.ntry

    # Allocate memory
    # Gout: real frequency Green's function, G(ω).
    # Gᵣ: imaginary frequency Green's function, G(iωₙ)
    ngrid, _ = size(SC.Λ)
    Gout = zeros(Complex{T}, nmesh)
    Gᵣ = zeros(T, ngrid)

    # Choose the best solution
    if method == "best"
        # The χ² of the best solution should be the smallest.
        p = argmin(SC.χ²ᵥ)
        println("Best solution: try = $p -> [χ² = $(SC.χ²ᵥ[p])]")
        #
        # Calculate G(ω)
        Gout = calc_green(p, SC, true, alg)
        #
        # Calculate G(iωₙ)
        Gᵣ = calc_green(p, SC, false, alg)
        #
        # Collect the `good` solutions and calculate their average.
        return -imag.(Gout) / π, (SC.fmesh[SC.Pᵥ[p]], SC.Aᵥ[p])
    else
        # Calculate the median of SC.χ²ᵥ
        chi2_med = median(SC.χ²ᵥ)
        chi2_ave = mean(SC.χ²ᵥ)

        # Determine the αgood parameter, which is used to filter the
        # calculated spectra.
        αgood = T(1.2)
        if count(x -> x < chi2_med / αgood, SC.χ²ᵥ) ≤ ntry / 10
            αgood = T(1)
        end

        # Go through all the solutions
        c = T(0) # A counter
        passed = I[]
        for i in 1:ntry
            if SC.χ²ᵥ[i] < chi2_med / αgood
                # Calculate and accumulate G(ω)
                G = calc_green(i, SC, true, alg)
                @. Gout = Gout + G
                #
                # Calculate and accumulate G(iωₙ)
                G = calc_green(i, SC, false, alg)
                @. Gᵣ = Gᵣ + G
                #
                # Increase the counter
                c = c + T(1)
                append!(passed, i)
            end
        end
        #
        # Normalize the final results
        @. Gout = Gout / c
        @. Gᵣ = Gᵣ / c
        println("Mean value of χ²: $(chi2_ave)")
        println("Median value of χ²: $(chi2_med)")
        println("Accumulate $(round(Int,c)) solutions to get the spectral density")
        return -imag.(Gout) / π
    end
end

#=
### *Core Algorithms*
=#

"""
    sample!(
        t::I,
        MC::StochPXMC{I},
        SE::StochPXElement{I, T},
        SC::StochPXContext{I, T},
        alg::SPX
    ) where {I<:Int, T<:Real}

Try to search the configuration space to locate the minimum by using the
simulated annealing algorithm. Here, `t` means the t-th attempt.

### Arguments
* t -> Counter for attempts.
* MC -> A StochPXMC struct.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.

### Returns
N/A
"""
function sample!(t::I,
                 MC::StochPXMC{I},
                 SE::StochPXElement{I,T},
                 SC::StochPXContext{I,T},
                 alg::SPX) where {I<:Int,T<:Real}
    # Try to change positions of poles
    if rand(MC.rng) < 0.5
        if rand(MC.rng) < 0.9
            try_move_s!(t, MC, SE, SC, alg)
        else
            try_move_p!(t, MC, SE, SC, alg)
        end
        # Try to change amplitudes of poles
    else
        if rand(MC.rng) < 0.5
            try_move_a!(t, MC, SE, SC, alg)
        else
            try_move_x!(t, MC, SE, SC, alg)
        end
    end
end

"""
    measure!(t::I, SE::StochPXElement{I, T}, SC::StochPXContext{I, T}) where {I<:Int, T<:Real}

Store Monte Carlo field configurations (positions, amplitudes, and signs
of many poles) for the `t`-th attempt. In other words, the current field
configuration (recorded in `SE`) will be saved in `SC`.

Note that not all configurations for the `t`-th attempt will be saved.
Only the solution that exhibits the smallest χ² will be saved. For the
`t`-th attempt, the StochPX solver will do `nstep` Monte Carlo updates.
It will calculate all χ², and try to figure out the smallest one. Then
the corresponding configuration (solution) will be saved.

### Arguments
* t -> Counter for the attemps.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.

### Returns
N/A
"""
function measure!(t::I, SE::StochPXElement{I,T},
                  SC::StochPXContext{I,T}) where {I<:Int,T<:Real}
    SC.χ²ᵥ[t] = SC.χ²
    #
    @. SC.Pᵥ[t] = SE.P
    @. SC.Aᵥ[t] = SE.A
    @. SC.𝕊ᵥ[t] = SE.𝕊
end

#=
### *Service Functions*
=#

"""
    init_mc(alg::SPX)

Try to create a StochPXMC struct. Some counters for Monte Carlo updates
are initialized here.

### Arguments
* S -> A StochPXSolver struct.

### Returns
* MC -> A StochPXMC struct.

See also: [`StochPXMC`](@ref).
"""
function init_mc(alg::SPX)
    seed = rand(1:100000000)
    rng = MersenneTwister(seed)
    #
    Sacc = 0
    Stry = 0
    #
    Pacc = 0
    Ptry = 0
    #
    Aacc = 0
    Atry = 0
    #
    Xacc = 0
    Xtry = 0
    #
    MC = StochPXMC(rng, Sacc, Stry, Pacc, Ptry, Aacc, Atry, Xacc, Xtry)

    return MC
end

"""
    init_element(
        alg::SPX,
        rng::AbstractRNG,
        T::Type{<:Real}
    ) where {T<:Real}

Randomize the configurations for future Monte Carlo sampling. It will
return a StochPXElement struct.

### Arguments
* alg -> A SPX struct.
* rng -> Random number generator.
* T -> Type of the real number.
### Returns
* SE -> A StochPXElement struct.

See also: [`StochPXElement`](@ref).
"""
function init_element(alg::SPX,
                      rng::AbstractRNG,
                      T::Type{<:Real})
    npole = alg.npole
    # Initialize P, A, and 𝕊
    P = rand(rng, collect(1:alg.nfine), npole)
    A = rand(rng, T, npole)
    𝕊 = ones(T, npole)

    # We have to make sure ∑ᵢ Aᵢ = 1
    s = sum(A)
    @. A = A / s

    SE = StochPXElement(abs.(P), A, 𝕊)

    return SE
end

"""
    init_context(
        alg::SPX,
        SE::StochPXElement{I, T},
        wn::Vector{T},
        fmesh::Vector{T},
        Gᵥ::Vector{T}
    ) where {I<:Int, T<:Real}

Try to initialize the key members of a StochPXContext struct. It will try
to return some key variables, which should be used to construct the
StochPXContext struct.

### Arguments
* alg -> A SPX struct.
* SE    -> A StochPXElement struct.
* wn    -> Grid for input correlator.
* fmesh -> Fine mesh in [wmin, wmax], used to build the kernel matrix Λ.
* Gᵥ    -> Preprocessed input correlator.

### Returns
* Gᵧ  -> Reconstructed correlator.
* Λ   -> Precomputed kernel matrix.
* Θ   -> Artificial inverse temperature.
* χ²  -> Current goodness-of-fit functional.
* χ²ᵥ -> Vector of goodness-of-fit functional.
* Pᵥ  -> Vector of poles' positions.
* Aᵥ  -> Vector of poles' amplitudes.
* 𝕊ᵥ  -> Vector of poles' signs.

See also: [`StochPXContext`](@ref).
"""
function init_context(alg::SPX,
                      SE::StochPXElement{I,T},
                      wn::Vector{T},
                      fmesh::Vector{T},
                      Gᵥ::Vector{T}) where {I<:Int,T<:Real}
    # Extract some parameters
    ntry = alg.ntry
    npole = alg.npole

    # Prepare the kernel matrix Λ. It is used to speed up the simulation.
    # Note that Λ depends on the type of kernel.
    Λ = calc_lambda(wn, fmesh)

    # We have to make sure that the starting Gᵧ and χ² are consistent with
    # the current Monte Carlo configuration fields.
    Gᵧ = calc_green(SE.P, SE.A, SE.𝕊, Λ)
    χ² = calc_chi2(Gᵧ, Gᵥ)

    # χ²ᵥ is initialized by a large number. Later it will be updated by
    # the smallest χ² during the simulation.
    χ²ᵥ = zeros(T, ntry)
    @. χ²ᵥ = T(1e10)

    # P, A, and 𝕊 should be always compatible with χ². They are updated
    # in the `measure!()` function.
    Pᵥ = Vector{I}[]
    Aᵥ = Vector{T}[]
    𝕊ᵥ = Vector{T}[]
    #
    for _ in 1:ntry
        push!(Pᵥ, ones(I, npole))
        push!(Aᵥ, zeros(T, npole))
        push!(𝕊ᵥ, zeros(T, npole))
    end

    return Gᵧ, Λ, T(alg.theta), χ², χ²ᵥ, Pᵥ, Aᵥ, 𝕊ᵥ
end

"""
    reset_mc!(MC::StochPXMC)

Reset the counters in StochPXMC struct.

### Arguments
* MC -> A StochPXMC struct.

### Returns
N/A
"""
function reset_mc!(MC::StochPXMC)
    MC.Sacc = 0
    MC.Stry = 0
    #
    MC.Pacc = 0
    MC.Ptry = 0
    #
    MC.Aacc = 0
    MC.Atry = 0
    #
    MC.Xacc = 0
    return MC.Xtry = 0
end

"""
    reset_element!(
        rng::AbstractRNG,
        SE::StochPXElement,
        alg::SPX
    )

Reset the Monte Carlo field configurations (i.e. positions and amplitudes
of the poles). Note that the signs of the poles should not be changed.
Be careful, the corresponding χ² (goodness-of-fit functional) and Gᵧ
(generated correlator) will be updated in the `reset_context!()` function.

### Arguments
* rng   -> Random number generator.
* allow -> Allowed positions for the poles.
* SE -> A StochPXElement struct.

### Returns
N/A
"""
function reset_element!(rng::AbstractRNG,
                        SE::StochPXElement{I,T},
                        alg::SPX) where {I<:Int,T<:Real}
    npole = alg.npole
    allow = collect(1:alg.nfine)

    # How many poles should be changed
    if npole ≤ 5
        if 4 ≤ npole ≤ 5
            nselect = 2
        else
            nselect = 1
        end
    else
        nselect = npole ÷ 5
    end
    @assert nselect ≤ npole

    # Which poles should be changed
    selected = rand(rng, 1:npole, nselect)
    unique!(selected)
    nselect = length(selected)

    # Change poles' positions
    if rand(rng) < 0.5
        P = rand(rng, allow, nselect)
        @. SE.P[selected] = P
        # Change poles' amplitudes
    else
        A₁ = SE.A[selected]
        s₁ = sum(A₁)
        #
        A₂ = rand(rng, T, nselect)
        s₂ = sum(A₂)
        @. A₂ = A₂ / s₂ * s₁
        #
        @. SE.A[selected] = A₂
    end
end

"""
    reset_context!(t::I, SE::StochPXElement{I, T}, SC::StochPXContext{I, T}, alg::SPX) where {I<:Int, T<:Real}

Recalculate imaginary frequency Green's function Gᵧ and goodness-of-fit
functional χ² by new Monte Carlo field configurations for the `t`-th
attempts. They must be consistent with each other.

Some key variables in `SC` are also updated as well. Perhaps we should
develop a smart algorhtm to update Θ here.

### Arguments
* t -> Counter for attempts.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
N/A
"""
function reset_context!(t::I, SE::StochPXElement{I,T}, SC::StochPXContext{I,T},
                        alg::SPX) where {I<:Int,T<:Real}
    SC.Θ = alg.theta
    SC.Gᵧ = calc_green(SE.P, SE.A, SE.𝕊, SC.Λ)
    SC.χ² = calc_chi2(SC.Gᵧ, SC.Gᵥ)
    return SC.χ²ᵥ[t] = T(1e10)
end

"""
    calc_lambda(wn::Vector{T}, fmesh::Vector{T}) where {T<:Real}

Precompute the kernel matrix Λ (Λ ≡ 1 / (iωₙ - ϵ)). It is a service
function and is for the fermionic systems.

### Arguments
* wn    -> Imaginary axis grid for input data.
* fmesh -> Very dense mesh in [wmin, wmax].

### Returns
* Λ -> The kernel matrix, a 2D array.
"""
function calc_lambda(wn::Vector{T}, fmesh::Vector{T}) where {T<:Real}
    ngrid = length(wn)
    nfine = length(fmesh)

    _Λ = zeros(Complex{T}, ngrid, nfine)
    #
    for i in eachindex(wn)
        iωₙ = im * wn[i]
        for j in eachindex(fmesh)
            _Λ[i, j] = 1 / (iωₙ - fmesh[j])
        end
    end
    #
    Λ = vcat(real(_Λ), imag(_Λ))

    return Λ
end

"""
    calc_green(t::I, SC::StochPXContext{I, T}, real_axis::Bool) where {I<:Int, T<:Real}

Reconstruct Green's function at imaginary axis or real axis by using the
pole expansion. It is a driver function. If `real_axis = true`, it will
returns G(ω), or else G(iωₙ).

### Arguments
* t -> Index of the current attempt.
* SC -> A StochPXContext struct.
* real_axis -> Working at real axis (true) or imaginary axis (false)?

### Returns
* G -> Reconstructed Green's function, G(ω) or G(iωₙ).
"""
function calc_green(t::I, SC::StochPXContext{I,T}, real_axis::Bool,
                    alg::SPX) where {I<:Int,T<:Real}
    @assert t ≤ alg.ntry
    # Calculate G(iωₙ)
    if real_axis == false
        return calc_green(SC.Pᵥ[t], SC.Aᵥ[t], SC.𝕊ᵥ[t], SC.Λ)
    end
    return calc_green(SC.Pᵥ[t], SC.Aᵥ[t], SC.𝕊ᵥ[t], SC.mesh.mesh, SC.fmesh, T(alg.eta))
end

"""
    calc_green(
        P::Vector{I},
        A::Vector{T},
        𝕊::Vector{T},
        Λ::Array{T,2}
    ) where {I<:Int, T<:Real}

Reconstruct Green's function at imaginary axis by the pole expansion.

### Arguments
* P -> Positions of poles.
* A -> Amplitudes of poles.
* 𝕊 -> Signs of poles.
* Λ -> Kernel matrix Λ.
* alg -> A SPX struct.

### Returns
* G -> Reconstructed Green's function, G(iωₙ).
"""
function calc_green(P::Vector{I},
                    A::Vector{T},
                    𝕊::Vector{T},
                    Λ::Array{T,2}) where {I<:Int,T<:Real}
    # Note that here `ngrid` is equal to 2 × ngrid sometimes.
    ngrid, _ = size(Λ)

    G = zeros(T, ngrid)
    for i in 1:ngrid
        G[i] = dot(A .* 𝕊, Λ[i, P])
    end

    return G
end

"""
    calc_green(
        P::Vector{I},
        A::Vector{T},
        𝕊::Vector{T},
        mesh::Vector{T},
        fmesh::Vector{T},
        η::T
    ) where {I<:Int, T<:Real}

Reconstruct Green's function at real axis by the pole expansion. It is
for the fermionic systems only.

### Arguments
* P     -> Positions of poles.
* A     -> Amplitudes of poles.
* 𝕊     -> Signs of poles.
* mesh  -> Real frequency mesh for spectral functions.
* fmesh -> Very dense real frequency mesh for poles.
* η -> Imaginary time step.

### Returns
* G -> Retarded Green's function, G(ω).
"""
function calc_green(P::Vector{I},
                    A::Vector{T},
                    𝕊::Vector{T},
                    mesh::Vector{T},
                    fmesh::Vector{T},
                    η::T) where {I<:Int,T<:Real}
    nmesh = length(mesh)

    iωₙ = mesh .+ im * η
    G = zeros(Complex{T}, nmesh)
    for i in eachindex(mesh)
        G[i] = sum(@. (A * 𝕊) / (iωₙ[i] - fmesh[P]))
    end

    return G
end

"""
    calc_chi2(Gₙ::Vector{T}, Gᵥ::Vector{T}) where {T<:Real}

Try to calculate the goodness-of-fit function (i.e, χ²), which measures
the distance between input and regenerated correlators.

### Arguments
* Gₙ -> Reconstructed Green's function.
* Gᵥ -> Original Green's function.

### Returns
* chi2 -> Goodness-of-fit functional, χ².

See also: [`calc_green`](@ref).
"""
function calc_chi2(Gₙ::Vector{T}, Gᵥ::Vector{T}) where {T<:Real}
    ΔG = Gₙ - Gᵥ
    return dot(ΔG, ΔG)
end

"""
    try_move_s!(
        t::I,
        MC::StochPXMC{I},
        SE::StochPXElement{I, T},
        SC::StochPXContext{I, T},
        alg::SPX
    ) where {I<:Int, T<:Real}

Change the position of one randomly selected pole.

### Arguments
* t -> Counter of attempts.
* MC -> A StochPXMC struct.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
N/A

See also: [`try_move_p`](@ref).
"""
function try_move_s!(t::I,
                     MC::StochPXMC{I},
                     SE::StochPXElement{I,T},
                     SC::StochPXContext{I,T},
                     alg::SPX) where {I<:Int,T<:Real}
    # Get parameters
    ngrid = length(SC.Gᵥ)
    nfine = alg.nfine
    npole = alg.npole
    move_window = nfine ÷ 100

    # It is used to save the change of Green's function
    δG = zeros(T, ngrid)
    Gₙ = zeros(T, ngrid)

    # Try to go through each pole
    for _ in 1:npole

        # Select one pole randomly
        s = rand(MC.rng, 1:npole)

        # Try to change position of the s pole
        Aₛ = SE.A[s]
        𝕊ₛ = SE.𝕊[s]
        #
        δP = rand(MC.rng, 1:move_window)
        #
        P₁ = SE.P[s]
        P₂ = P₁
        if rand(MC.rng) > 0.5
            P₂ = P₁ + δP
        else
            P₂ = P₁ - δP
        end
        #
        if 𝕊ₛ > 0.0
            !(+P₂ in SC.allow) && continue
        else
            !(-P₂ in SC.allow) && continue
        end

        # Calculate change of Green's function
        Λ₁ = view(SC.Λ, :, P₁)
        Λ₂ = view(SC.Λ, :, P₂)
        @. δG = 𝕊ₛ * Aₛ * (Λ₂ - Λ₁)

        # Calculate new Green's function and goodness-of-fit function
        @. Gₙ = δG + SC.Gᵧ
        χ² = calc_chi2(Gₙ, SC.Gᵥ)
        δχ² = χ² - SC.χ²

        # Simulated annealing algorithm
        MC.Stry = MC.Stry + 1
        if δχ² < 0 || min(1, exp(-δχ² * SC.Θ)) > rand(MC.rng)
            # Update Monte Carlo configuration
            SE.P[s] = P₂

            # Update reconstructed Green's function
            @. SC.Gᵧ = Gₙ

            # Update goodness-of-fit function
            SC.χ² = χ²

            # Update Monte Carlo counter
            MC.Sacc = MC.Sacc + 1

            # Save optimal solution
            if SC.χ² < SC.χ²ᵥ[t]
                measure!(t, SE, SC)
            end
        end
    end
end

"""
    try_move_p!(
        t::I,
        MC::StochPXMC{I},
        SE::StochPXElement{I, T},
        SC::StochPXContext{I, T},
        alg::SPX
    ) where {I<:Int, T<:Real}

Change the positions of two randomly selected poles.

### Arguments
* t -> Counter of attempts.
* MC -> A StochPXMC struct.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
N/A

See also: [`try_move_s!`](@ref).
"""
function try_move_p!(t::I,
                     MC::StochPXMC{I},
                     SE::StochPXElement{I,T},
                     SC::StochPXContext{I,T},
                     alg::SPX) where {I<:Int,T<:Real}
    # Get parameters
    ngrid = length(SC.Gᵥ)
    npole = alg.npole

    # Sanity check
    if npole == 1
        return
    end

    # It is used to save the change of Green's function
    δG = zeros(T, ngrid)
    Gₙ = zeros(T, ngrid)

    # Try to go through each pole
    for _ in 1:npole

        # Select two poles randomly
        # The two poles should not be the same.
        s₁ = 1
        s₂ = 1
        while s₁ == s₂
            s₁ = rand(MC.rng, 1:npole)
            s₂ = rand(MC.rng, 1:npole)
        end

        # Try to change position of the s₁ pole
        A₁ = SE.A[s₁]
        𝕊₁ = SE.𝕊[s₁]
        P₁ = SE.P[s₁]
        P₃ = P₁
        while P₃ == P₁ || sign(P₃) != sign(𝕊₁)
            P₃ = rand(MC.rng, SC.allow)
        end
        P₃ = abs(P₃)
        #
        # Try to change position of the s₂ pole
        A₂ = SE.A[s₂]
        𝕊₂ = SE.𝕊[s₂]
        P₂ = SE.P[s₂]
        P₄ = P₂
        while P₄ == P₂ || sign(P₄) != sign(𝕊₂)
            P₄ = rand(MC.rng, SC.allow)
        end
        P₄ = abs(P₄)

        # Calculate change of Green's function
        Λ₁ = view(SC.Λ, :, P₁)
        Λ₂ = view(SC.Λ, :, P₂)
        Λ₃ = view(SC.Λ, :, P₃)
        Λ₄ = view(SC.Λ, :, P₄)
        @. δG = 𝕊₁ * A₁ * (Λ₃ - Λ₁) + 𝕊₂ * A₂ * (Λ₄ - Λ₂)

        # Calculate new Green's function and goodness-of-fit function
        @. Gₙ = δG + SC.Gᵧ
        χ² = calc_chi2(Gₙ, SC.Gᵥ)
        δχ² = χ² - SC.χ²

        # Simulated annealing algorithm
        MC.Ptry = MC.Ptry + 1
        if δχ² < 0 || min(1.0, exp(-δχ² * SC.Θ)) > rand(MC.rng)
            # Update Monte Carlo configuration
            SE.P[s₁] = P₃
            SE.P[s₂] = P₄

            # Update reconstructed Green's function
            @. SC.Gᵧ = Gₙ

            # Update goodness-of-fit function
            SC.χ² = χ²

            # Update Monte Carlo counter
            MC.Pacc = MC.Pacc + 1

            # Save optimal solution
            if SC.χ² < SC.χ²ᵥ[t]
                measure!(t, SE, SC)
            end
        end
    end
end

"""
    try_move_a!(
        t::I,
        MC::StochPXMC{I},
        SE::StochPXElement{I, T},
        SC::StochPXContext{I, T},
        alg::SPX
    ) where {I<:Int, T<:Real}

Change the amplitudes of two randomly selected poles.

### Arguments
* t -> Counter of attempts.
* MC -> A StochPXMC struct.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
N/A

See also: [`try_move_x`](@ref).
"""
function try_move_a!(t::I,
                     MC::StochPXMC{I},
                     SE::StochPXElement{I,T},
                     SC::StochPXContext{I,T},
                     alg::SPX) where {I<:Int,T<:Real}
    # Get parameters
    ngrid = length(SC.Gᵥ)
    npole = alg.npole

    # Sanity check
    if npole == 1
        return
    end

    # It is used to save the change of Green's function
    δG = zeros(T, ngrid)
    Gₙ = zeros(T, ngrid)

    # Try to go through each pole
    for _ in 1:npole

        # Select two poles randomly
        # The two poles should not be the same.
        s₁ = 1
        s₂ = 1
        while s₁ == s₂
            s₁ = rand(MC.rng, 1:npole)
            s₂ = rand(MC.rng, 1:npole)
        end

        # Try to change amplitudes of the two poles, but their sum is kept.
        P₁ = SE.P[s₁]
        P₂ = SE.P[s₂]
        A₁ = SE.A[s₁]
        A₂ = SE.A[s₂]
        A₃ = T(0)
        A₄ = T(0)
        𝕊₁ = SE.𝕊[s₁]
        𝕊₂ = SE.𝕊[s₂]

        if 𝕊₁ == 𝕊₂
            while true
                δA = rand(MC.rng) * (A₁ + A₂) - A₁
                A₃ = A₁ + δA
                A₄ = A₂ - δA

                if 1.0 > A₃ > 0.0 && 1.0 > A₄ > 0.0
                    break
                end
            end
        else
            while true
                _δA = rand(MC.rng) * (A₁ + A₂) - A₁
                δA = rand(MC.rng) > 0.5 ? _δA * (+T(1)) : _δA * (-T(1))
                A₃ = (𝕊₁ * A₁ + δA) / 𝕊₁
                A₄ = (𝕊₂ * A₂ - δA) / 𝕊₂

                if 1.0 > A₃ > 0.0 && 1.0 > A₄ > 0.0
                    break
                end
            end
        end

        # Calculate change of Green's function
        Λ₁ = view(SC.Λ, :, P₁)
        Λ₂ = view(SC.Λ, :, P₂)
        @. δG = 𝕊₁ * (A₃ - A₁) * Λ₁ + 𝕊₂ * (A₄ - A₂) * Λ₂

        # Calculate new Green's function and goodness-of-fit function
        @. Gₙ = δG + SC.Gᵧ
        χ² = calc_chi2(Gₙ, SC.Gᵥ)
        δχ² = χ² - SC.χ²

        # Simulated annealing algorithm
        MC.Atry = MC.Atry + 1
        if δχ² < 0 || min(1.0, exp(-δχ² * SC.Θ)) > rand(MC.rng)
            # Update Monte Carlo configuration
            SE.A[s₁] = A₃
            SE.A[s₂] = A₄

            # Update reconstructed Green's function
            @. SC.Gᵧ = Gₙ

            # Update goodness-of-fit function
            SC.χ² = χ²

            # Update Monte Carlo counter
            MC.Aacc = MC.Aacc + 1

            # Save optimal solution
            if SC.χ² < SC.χ²ᵥ[t]
                measure!(t, SE, SC)
            end
        end
    end
end

"""
    try_move_x!(
        t::I,
        MC::StochPXMC{I},
        SE::StochPXElement{I, T},
        SC::StochPXContext{I, T},
        alg::SPX
    ) where {I<:Int, T<:Real}

Exchange the amplitudes of two randomly selected poles.

### Arguments
* t -> Counter of attempts.
* MC -> A StochPXMC struct.
* SE -> A StochPXElement struct.
* SC -> A StochPXContext struct.
* alg -> A SPX struct.

### Returns
N/A

See also: [`try_move_a`](@ref).
"""
function try_move_x!(t::I,
                     MC::StochPXMC{I},
                     SE::StochPXElement{I,T},
                     SC::StochPXContext{I,T},
                     alg::SPX) where {I<:Int,T<:Real}
    # Get parameters
    ngrid = length(SC.Gᵥ)
    npole = alg.npole

    if npole == 1
        return
    end

    # It is used to save the change of Green's function
    δG = zeros(T, ngrid)
    Gₙ = zeros(T, ngrid)

    # Try to go through each pole
    for _ in 1:npole

        # Select two poles randomly
        # The positions of the two poles are different,
        # but their signs should be the same.
        s₁ = 1
        s₂ = 1
        while (s₁ == s₂) || (SE.𝕊[s₁] != SE.𝕊[s₂])
            s₁ = rand(MC.rng, 1:npole)
            s₂ = rand(MC.rng, 1:npole)
        end

        # Try to swap amplitudes of the two poles, but their sum is kept.
        P₁ = SE.P[s₁]
        P₂ = SE.P[s₂]
        A₁ = SE.A[s₁]
        A₂ = SE.A[s₂]
        A₃ = A₂
        A₄ = A₁
        𝕊₁ = SE.𝕊[s₁]
        𝕊₂ = SE.𝕊[s₂]

        # Calculate change of Green's function
        Λ₁ = view(SC.Λ, :, P₁)
        Λ₂ = view(SC.Λ, :, P₂)
        @. δG = 𝕊₁ * (A₃ - A₁) * Λ₁ + 𝕊₂ * (A₄ - A₂) * Λ₂

        # Calculate new Green's function and goodness-of-fit function
        @. Gₙ = δG + SC.Gᵧ
        χ² = calc_chi2(Gₙ, SC.Gᵥ)
        δχ² = χ² - SC.χ²

        # Simulated annealing algorithm
        MC.Xtry = MC.Xtry + 1
        if δχ² < 0 || min(1.0, exp(-δχ² * SC.Θ)) > rand(MC.rng)
            # Update Monte Carlo configuration
            SE.A[s₁] = A₃
            SE.A[s₂] = A₄

            # Update reconstructed Green's function
            @. SC.Gᵧ = Gₙ

            # Update goodness-of-fit function
            SC.χ² = χ²

            # Update Monte Carlo counter
            MC.Xacc = MC.Xacc + 1

            # Save optimal solution
            if SC.χ² < SC.χ²ᵥ[t]
                measure!(t, SE, SC)
            end
        end
    end
end

# solve differentiation
function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::SPX;
                   diffonly::Bool=false) where {T<:Real}
    if ctx.spt isa Cont
        return Adiff(GFV, ctx, alg; ns=true, diffonly=diffonly)
    elseif ctx.spt isa Delta
        return pγdiff(GFV, ctx, alg; ns=true)
    else
        error("Unsupported spectral function type")
    end
end
