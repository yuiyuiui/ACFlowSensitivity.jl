#
# Project : Gardenia
# Source  : som.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/10/01
#

#=
### *Customized Structs* : *StochOM Solver*
=#

"""
    Box

Rectangle. The field configuration consists of many boxes. They exhibit
various areas (width × height). We use the Metropolis important sampling
algorithm to sample them and evaluate their contributions to the spectrum.

### Members
* h -> Height of the box.
* w -> Width of the box.
* c -> Position of the box.
"""
mutable struct Box{T<:Real}
    h::T
    w::T
    c::T
end

"""
    StochOMMC

Mutable struct. It is used within the StochOM solver. It includes random
number generator and some counters.

Because the StochOM solver supports many Monte Carlo updates, so `Macc`
and `Mtry` are vectors.

### Members
* rng  -> Random number generator.
* Macc -> Counter for move operation (accepted).
* Mtry -> Counter for move operation (tried).

See also: [`StochOMSolver`](@ref).
"""
mutable struct StochOMMC{I<:Int}
    rng::AbstractRNG
    Macc::Vector{I}
    Mtry::Vector{I}
end

"""
    StochOMElement

Mutable struct. It is used to record the field configurations, which will
be sampled by Monte Carlo sweeping procedure.

### Members
* C -> Field configuration.
* Λ -> Contributions of the field configuration to the correlator.
* G -> Reproduced correlator.
* Δ -> Difference between reproduced and raw correlators.
"""
mutable struct StochOMElement{T<:Real}
    C::Vector{Box{T}}
    Λ::Array{T,2}
    G::Vector{T}
    Δ::T
end

"""
    StochOMContext

Mutable struct. It is used within the StochOM solver only.

### Members
* Gᵥ    -> Input data for correlator.
* σ¹    -> Actually 1 / σ¹.
* grid  -> Grid for input data.
* mesh  -> Mesh for output spectrum.
* Cᵥ    -> It is used to record the field configurations for all attempts.
* Δᵥ    -> It is used to record the errors for all attempts.
* 𝕊ᵥ    -> It is used to interpolate the Λ functions.
"""
mutable struct StochOMContext{T<:Real}
    Gᵥ::Vector{T}
    σ¹::T
    wn::Vector{T}
    mesh::Vector{T}
    mesh_weight::Vector{T}
    Cᵥ::Vector{Vector{Box{T}}}
    Δᵥ::Vector{T}
    𝕊ᵥ::Vector{CubicSplineInterpolation}
end

#=
### *Global Drivers*
=#

"""
    solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::SOM)

Solve the analytic continuation problem by the stochastic optimization
method. This solver requires a lot of computational resources to get
reasonable results. It is suitable for both Matsubara and imaginary
time correlators. It is the driver for the StochOM solver.

If the input correlators are bosonic, this solver will return A(ω) / ω
via `Aout`, instead of A(ω). At this time, `Aout` is not compatible with
`Gout`. If the input correlators are fermionic, this solver will return
A(ω) in `Aout`. Now it is compatible with `Gout`. These behaviors are just
similar to the MaxEnt, StochAC, and StochSK solvers.

Now the StochOM solver supports both continuous and δ-like spectra.

### Arguments
* GFV -> A vector of complex numbers, containing the imaginary-time Green's function.
* ctx -> A CtxData struct, containing the context data.
* alg -> A SOM struct, containing the algorithm parameters.

### Returns
* mesh -> Real frequency mesh, ω.
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::SOM) where {T<:Real}
    # Initialize counters for Monte Carlo engine
    MC = init_mc(alg)
    println("Create infrastructure for Monte Carlo sampling")

    # Prepare some key variables
    Cᵥ, Δᵥ, 𝕊ᵥ = init_context(alg, ctx)
    SC = StochOMContext(vcat(real(GFV), imag(GFV)), 1/ctx.σ, ctx.wn, ctx.mesh,
                        ctx.mesh_weight, Cᵥ, Δᵥ, 𝕊ᵥ)
    println("Initialize context for the StochOM solver")

    Aout = run!(MC, SC, alg)

    if ctx.spt isa Cont
        return SC.mesh, Aout
    elseif ctx.spt isa Delta
        p = ctx.mesh[find_peaks(ctx.mesh, Aout, ctx.fp_mp; wind=ctx.fp_ww)]
        γ = ones(T, length(p)) ./ length(p)
        return SC.mesh, Aout, (p, γ)
    else
        error("Unsupported spectral function type")
    end
end

"""
    run!(MC::StochOMMC{I}, SC::StochOMContext{T}, alg::SOM) where {T<:Real, I<:Int}

Perform stochastic optimization simulation, sequential version.

### Arguments
* MC -> A StochOMMC struct.
* SC -> A StochOMContext struct.
* alg -> A SOM struct, containing the algorithm parameters.
### Returns
* Aout -> Spectral function, A(ω).
"""
function run!(MC::StochOMMC{I}, SC::StochOMContext{T}, alg::SOM) where {T<:Real,I<:Int}
    # Setup essential parameters
    ntry = alg.ntry
    nstep = alg.nstep

    # Sample and collect data
    println("Start stochastic sampling...")
    for l in 1:ntry
        # Re-initialize the simulation
        SE = init_element(MC, SC, alg)

        # For each attempt, we should perform `nstep × N` Monte Carlo
        # updates, where `N` means length of the Markov chain.
        for _ in 1:nstep
            update!(MC, SE, SC, alg)
        end

        # Accumulate the data
        SC.Δᵥ[l] = SE.Δ
        SC.Cᵥ[l] = deepcopy(SE.C)

        # Show error function for the current attempt
        l%20==0 && @show l, SE.Δ
    end

    # Generate spectral density from Monte Carlo field configuration
    return average(SC, length(SC.mesh), alg)
end

"""
    average(SC::StochOMContext, nmesh::Int, alg::SOM)

Postprocess the collected results after the stochastic optimization
simulations. It will generate the spectral functions.

### Arguments
* SC -> A StochOMContext struct.

### Returns
* Aom -> Spectral function, A(ω).
"""
function average(SC::StochOMContext{T}, nmesh::Int, alg::SOM) where {T<:Real}
    ntry = alg.ntry

    # Calculate the median of SC.Δᵥ
    dev_ave = median(SC.Δᵥ)

    # Determine the αgood parameter, which is used to filter the
    # calculated spectra.
    αgood = T(1.2)
    if count(x -> x < dev_ave / αgood, SC.Δᵥ) == 0
        αgood = T(1)
    end

    # Accumulate the final spectrum
    Aom = zeros(T, nmesh)
    passed = Int[]
    for l in 1:ntry
        # Filter the reasonable spectra
        if SC.Δᵥ[l] < dev_ave / αgood
            # Generate the spectrum, and add it to Aom.
            for w in 1:nmesh
                _omega = SC.mesh[w]
                # Scan all boxes
                for r in 1:length(SC.Cᵥ[l])
                    R = SC.Cᵥ[l][r]
                    # Yes, this box contributes. The point, _omega, is
                    # covered by this box.
                    if R.c - T(0.5) * R.w ≤ _omega ≤ R.c + T(0.5) * R.w
                        Aom[w] = Aom[w] + R.h
                    end
                end
            end
            #
            # Record which spectrum is used
            append!(passed, l)
        end
    end

    # Normalize the spectrum
    Lgood = count(x -> x < dev_ave / αgood, SC.Δᵥ)
    @assert Lgood == length(passed)
    @. Aom = Aom / Lgood
    @show dev_ave, Lgood

    return Aom
end

#=
### *Core Algorithms*
=#

"""
    update!(MC::StochOMMC{I}, SE::StochOMElement{T}, SC::StochOMContext{T}, alg::SOM) where {T<:Real, I<:Int}

Using the Metropolis algorithm to update the field configuration, i.e, a
collection of hundreds of boxes. Be careful, this function only updates
the Monte Carlo configurations (in other words, `SE`). It doesn't record
them. Measurements are done in `run()` and `prun()`. This is the reason
why this function is named as `update!()`, instead of `sample()`.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* alg -> A SOM struct, containing the algorithm parameters.
### Returns
N/A
"""
function update!(MC::StochOMMC{I}, SE::StochOMElement{T}, SC::StochOMContext{T},
                 alg::SOM) where {T<:Real,I<:Int}
    Tmax = I(100) # Length of the Markov chain
    nbox = alg.nbox

    ST = deepcopy(SE)

    # The Markov chain is divided into two stages
    T1 = rand(MC.rng, 1:Tmax)
    d1 = rand(MC.rng, T)
    d2 = T(1) + rand(MC.rng, T)
    #
    # The first stage
    for _ in 1:T1
        update_type = rand(MC.rng, 1:7)

        if update_type == 1
            if 1 ≤ length(ST.C) ≤ nbox - 1
                try_insert(MC, ST, SC, d1, alg)
            end
            break

        elseif update_type == 2
            if length(ST.C) ≥ 2
                try_remove(MC, ST, SC, d1, alg)
            end
            break

        elseif update_type == 3
            if length(ST.C) ≥ 1
                try_shift(MC, ST, SC, d1, alg)
            end
            break

        elseif update_type == 4
            if length(ST.C) ≥ 1
                try_width(MC, ST, SC, d1, alg)
            end
            break

        elseif update_type == 5
            if length(ST.C) ≥ 2
                try_height(MC, ST, SC, d1, alg)
            end
            break

        elseif update_type == 6
            if 1 ≤ length(ST.C) ≤ nbox - 1
                try_split(MC, ST, SC, d1, alg)
            end
            break

        elseif update_type == 7
            if length(ST.C) ≥ 2
                try_merge(MC, ST, SC, d1, alg)
            end
            break
        end
    end
    #
    # The second stage
    for _ in (T1 + 1):Tmax
        update_type = rand(MC.rng, 1:7)

        if update_type == 1
            if 1 ≤ length(ST.C) ≤ nbox - 1
                try_insert(MC, ST, SC, d2, alg)
            end
            break

        elseif update_type == 2
            if length(ST.C) ≥ 2
                try_remove(MC, ST, SC, d2, alg)
            end
            break

        elseif update_type == 3
            if length(ST.C) ≥ 1
                try_shift(MC, ST, SC, d2, alg)
            end
            break

        elseif update_type == 4
            if length(ST.C) ≥ 1
                try_width(MC, ST, SC, d2, alg)
            end
            break

        elseif update_type == 5
            if length(ST.C) ≥ 2
                try_height(MC, ST, SC, d2, alg)
            end
            break

        elseif update_type == 6
            if 1 ≤ length(ST.C) ≤ nbox - 1
                try_split(MC, ST, SC, d2, alg)
            end
            break

        elseif update_type == 7
            if length(ST.C) ≥ 2
                try_merge(MC, ST, SC, d2, alg)
            end
            break
        end
    end

    if ST.Δ < SE.Δ
        SE.C = deepcopy(ST.C)
        SE.Λ .= ST.Λ
        SE.G .= ST.G
        SE.Δ = ST.Δ
    end
end

#=
### *Service Functions*
=#

"""
    init_mc(S::SOM)

Try to create a StochOMMC struct. Some counters for Monte Carlo updates
are initialized here.

### Arguments
* S -> A StochOMSolver struct.

### Returns
* MC -> A StochOMMC struct.

See also: [`StochOMMC`](@ref).
"""
function init_mc(S::SOM)
    seed = rand(1:100000000)
    rng = MersenneTwister(seed)
    Macc = zeros(Int, 7)
    Mtry = zeros(Int, 7)
    MC = StochOMMC(rng, Macc, Mtry)
    return MC
end

"""
    init_element(MC::StochOMMC{I}, SC::StochOMContext{T}, alg::SOM) where {T<:Real, I<:Int}

Try to initialize a StochOMElement struct. In other words, we should
randomize the configurations for future Monte Carlo sampling here.

### Arguments
* MC -> A StochOMMC struct.
* SC -> A StochOMContext struct.
* alg -> A SOM struct, containing the algorithm parameters.
### Returns
* SE -> A StochOMElement struct.

See also: [`StochOMElement`](@ref).
"""
function init_element(MC::StochOMMC{I}, SC::StochOMContext{T},
                      alg::SOM) where {T<:Real,I<:Int}
    wmin = SC.mesh[1]
    wmax = SC.mesh[end]
    nbox = alg.nbox
    sbox = T(alg.sbox)
    wbox = T(alg.wbox)

    # Generate weights randomly
    _Know = rand(MC.rng, 2:nbox)
    _weight = zeros(T, _Know)
    for i in 1:_Know
        _weight[i] = rand(MC.rng, T)
    end
    _weight[end] = T(1)

    # Sort weights, make sure the sum of weights is always 1
    sort!(_weight)
    weight = diff(_weight)
    insert!(weight, 1, _weight[1])
    sort!(weight)

    # Make sure that each weight is larger than sbox.
    plus_count = 1
    minus_count = _Know
    while weight[plus_count] < sbox
        while weight[minus_count] < 2 * sbox
            minus_count = minus_count - 1
            minus_count == 0 && error("nbox * sbox is too big")
        end
        weight[plus_count] = weight[plus_count] + sbox
        weight[minus_count] = weight[minus_count] - sbox
        plus_count = plus_count + 1
    end

    # Create some boxes with random c, w, and h.
    C = Box{T}[]
    Λ = zeros(T, length(SC.Gᵥ), nbox)
    Δ = T(0)
    #
    for k in 1:_Know
        c = wmin + wbox / 2 + (wmax - wmin - wbox) * rand(MC.rng, T)
        w = wbox + (min(2 * (c - wmin), 2 * (wmax - c)) - wbox) * rand(MC.rng, T)
        while !constraints(c - w/2, c + w/2)
            c = wmin + wbox / 2 + (wmax - wmin - wbox) * rand(MC.rng, T)
            w = wbox + (min(2 * (c - wmin), 2 * (wmax - c)) - wbox) * rand(MC.rng, T)
        end
        h = weight[k] / w
        R = Box(h, w, c)
        push!(C, R)
        Λ[:, k] .= eval_lambda(R, SC.wn, SC.𝕊ᵥ)
    end
    #
    # Calculate Green's function and relative error using boxes
    G = calc_green(Λ, _Know)
    Δ = calc_error(G, SC.Gᵥ, SC.σ¹)

    return StochOMElement(C, Λ, G, Δ)
end

"""
    init_context(alg::SOM, ctx::CtxData{T}) where {T<:Real}

Try to initialize the key members of a StochOMContext struct.

### Arguments
* alg -> A SOM struct, containing the algorithm parameters.
* ctx -> A CtxData struct, containing the context data.

### Returns
* Cᵥ -> Field configurations for all attempts.
* Δᵥ -> Errors for all attempts.
* 𝕊ᵥ -> Interpolators for the Λ functions.

See also: [`StochOMContext`](@ref).
"""
function init_context(alg::SOM, ctx::CtxData{T}) where {T<:Real}
    ntry = alg.ntry
    nbox = alg.nbox
    ngrid = length(ctx.wn)

    # Initialize errors
    Δᵥ = zeros(T, ntry)

    # Initialize field configurations (boxes)
    Cᵥ = Vector{Vector{Box{T}}}()
    for _ in 1:ntry
        C = Box{T}[]
        for _ in 1:nbox
            push!(C, Box(T(0), T(0), T(0)))
        end
        push!(Cᵥ, C)
    end

    # Initialize interpolants 𝕊ᵥ
    # It is useful only when the input data is in imaginary time axis.
    𝕊ᵥ = Vector{CubicSplineInterpolation}(undef, ngrid)
    return Cᵥ, Δᵥ, 𝕊ᵥ
end

"""
    eval_lambda(
        r::Box,
        wn::Vector{T},
        𝕊::Vector{<:AbstractInterpolation}
    ) where {T<:Real}

Try to calculate the contribution of a given box `r` to the Λ function.
This function works for FermionicMatsubaraGrid only. Because there is an
analytic expression for this case, 𝕊 is useless.

Actually, 𝕊 is undefined here. See init_context().

### Arguments
* r    -> A box or rectangle.
* grid -> Imaginary axis grid for input data.
* 𝕊    -> An interpolant.

### Returns
* Λ -> Λ(iωₙ) function, 1D function.

See also: [`FermionicMatsubaraGrid`](@ref).
"""
function eval_lambda(r::Box,
                     wn::Vector{T},
                     𝕊::Vector{<:AbstractInterpolation}) where {T<:Real}
    # Get left and right boundaries of the given box
    e₁ = r.c - T(0.5) * r.w
    e₂ = r.c + T(0.5) * r.w

    # Evaluate Λ
    iwn = im * wn
    Λ = @. r.h * log((iwn - e₁) / (iwn - e₂))

    return vcat(real(Λ), imag(Λ))
end

"""
    calc_error(G::Vector{T}, Gᵥ::Vector{T}, σ¹::T) where {T<:Real}

Try to calculate χ². Here `Gᵥ` and `σ¹` denote the raw correlator and
related standard deviation. `G` means the reproduced correlator.

### Arguments
See above explanations.

### Returns
* Δ -> χ², distance between reconstructed and raw correlators.

See also: [`calc_green`](@ref).
"""
function calc_error(G::Vector{T}, Gᵥ::Vector{T}, σ¹::T) where {T<:Real}
    return sum(((G .- Gᵥ) .* σ¹) .^ T(2))
end

"""
    calc_green(Λ::Array{T,2}, nk::I) where {T<:Real, I<:Int}

Try to reconstruct the correlator via the field configuration. Now this
function is called by init_element(). But perhaps we can use it in last().

### Arguments
* Λ -> The Λ function. See above remarks.
* nk -> Current number of boxes.

### Returns
* G -> Reconstructed Green's function.

See also: [`calc_error`](@ref).
"""
function calc_green(Λ::Array{T,2}, nk::I) where {T<:Real,I<:Int}
    ngrid, nbox = size(Λ)
    @assert nk ≤ nbox

    G = zeros(T, ngrid)
    for k in 1:nk
        for g in 1:ngrid
            G[g] = G[g] + Λ[g, k]
        end
    end

    return G
end

"""
    constraints(e₁::T, e₂::T) where {T<:Real}

This function is used to judge whether a given box overlapes with the
forbidden zone. Here `e₁` and `e₂` denote the left and right boundaries
of the box.

### Arguments
See above explanations.

### Returns
* ex -> Boolean, whether a given box is valid.

"""
function constraints(e₁::T, e₂::T) where {T<:Real}
    # exclude = get_b("exclude")
    exclude = missing
    @assert e₁ ≤ e₂

    if !isa(exclude, Missing)
        for i in eachindex(exclude)
            if e₁ ≤ exclude[i][1] ≤ e₂ ≤ exclude[i][2]
                return false
            end
            if exclude[i][1] ≤ e₁ ≤ exclude[i][2] ≤ e₂
                return false
            end
            if exclude[i][1] ≤ e₁ ≤ e₂ ≤ exclude[i][2]
                return false
            end
            if e₁ ≤ exclude[i][1] ≤ exclude[i][2] ≤ e₂
                return false
            end
        end
    end

    return true
end

"""
    try_insert(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Insert a new box into the field configuration.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.

### Returns
N/A
"""
function try_insert(MC::StochOMMC{I},
                    SE::StochOMElement{T},
                    SC::StochOMContext{T},
                    dacc::T,
                    alg::SOM) where {T<:Real,I<:Int}
    sbox = T(alg.sbox)
    wbox = T(alg.wbox)
    wmin = SC.mesh[1]
    wmax = SC.mesh[end]
    csize = length(SE.C)

    # Choose a box randomly
    t = rand(MC.rng, 1:csize)

    # Check area of this box
    R = SE.C[t]
    if R.h * R.w ≤ T(2) * sbox
        return
    end

    # Determine minimum and maximum areas of the new box
    dx_min = sbox
    dx_max = R.h * R.w - sbox
    if dx_max ≤ dx_min
        return
    end

    # Determine parameters for the new box
    r₁ = rand(MC.rng, T)
    r₂ = rand(MC.rng, T)
    #
    c = (wmin + wbox / T(2)) + (wmax - wmin - wbox) * r₁
    #
    w_new_max = T(2) * min(wmax - c, c - wmin)
    dx = Pdx(dx_min, dx_max, MC.rng)
    #
    h = dx / w_new_max + (dx / wbox - dx / w_new_max) * r₂ # TODO: check this
    w = dx / h

    # Rnew will be used to update Box t, while Radd is the new box.
    if !constraints(c - w/2, c + w/2)
        return
    end
    Rnew = Box(R.h - dx / R.w, R.w, R.c)
    Radd = Box(h, w, c)

    # Calculate update for Λ
    G₁ = SE.Λ[:, t]
    G₂ = eval_lambda(Rnew, SC.wn, SC.𝕊ᵥ)
    G₃ = eval_lambda(Radd, SC.wn, SC.𝕊ᵥ)

    # Calculate new Δ function, it is actually the error function.
    Δ = calc_error(SE.G - G₁ + G₂ + G₃, SC.Gᵥ, SC.σ¹)

    # Apply the Metropolis algorithm
    if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
        # Update box t
        SE.C[t] = Rnew

        # Add new Box
        push!(SE.C, Radd)

        # Update Δ, G, and Λ.
        SE.Δ = Δ
        @. SE.G = SE.G - G₁ + G₂ + G₃
        @. SE.Λ[:, t] = G₂
        @. SE.Λ[:, csize+1] = G₃

        # Update the counter
        MC.Macc[1] = MC.Macc[1] + 1
    end

    # Update the counter
    return MC.Mtry[1] = MC.Mtry[1] + 1
end

"""
    try_remove(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Remove an old box from the field configuration.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.
* alg  -> A StochOM struct. It contains parameters for the algorithm.

### Returns
N/A
"""
function try_remove(MC::StochOMMC{I},
                    SE::StochOMElement{T},
                    SC::StochOMContext{T},
                    dacc::T,
                    alg::SOM) where {T<:Real,I<:Int}
    csize = length(SE.C)

    # Choose two boxes randomly
    # Box t₁ will be removed, while box t₂ will be modified.
    t₁ = rand(MC.rng, 1:csize)
    t₂ = rand(MC.rng, 1:csize)
    #
    while t₁ == t₂
        t₂ = rand(MC.rng, 1:csize)
    end
    #
    if t₁ < t₂
        t₁, t₂ = t₂, t₁
    end

    # Get box t₁ and box t₂
    R₁ = SE.C[t₁]
    R₂ = SE.C[t₂]
    Rₑ = SE.C[end]

    # Generate new box t₂
    dx = R₁.h * R₁.w
    R₂ₙ = Box(R₂.h + dx / R₂.w, R₂.w, R₂.c)

    # Calculate update for Λ
    G₁ = SE.Λ[:, t₁]
    G₂ = SE.Λ[:, t₂]
    Gₑ = SE.Λ[:, csize]
    G₂ₙ = eval_lambda(R₂ₙ, SC.wn, SC.𝕊ᵥ)

    # Calculate new Δ function, it is actually the error function.
    Δ = calc_error(SE.G - G₁ - G₂ + G₂ₙ, SC.Gᵥ, SC.σ¹)

    # Apply the Metropolis algorithm
    if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
        # Update box t₂
        SE.C[t₂] = R₂ₙ

        # Backup the last box in box t₁
        if t₁ < csize
            SE.C[t₁] = Rₑ
        end

        # Delete the last box, since its value has been stored in t₁.
        pop!(SE.C)

        # Update Δ, G, and Λ.
        SE.Δ = Δ
        @. SE.G = SE.G - G₁ - G₂ + G₂ₙ
        @. SE.Λ[:, t₂] = G₂ₙ
        if t₁ < csize
            @. SE.Λ[:, t₁] = Gₑ
        end

        # Update the counter
        MC.Macc[2] = MC.Macc[2] + 1
    end

    # Update the counter
    return MC.Mtry[2] = MC.Mtry[2] + 1
end

"""
    try_shift(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Change the position of given box in the field configuration.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.
* alg  -> A StochOM struct. It contains parameters for the algorithm.

### Returns
N/A
"""
function try_shift(MC::StochOMMC{I},
                   SE::StochOMElement{T},
                   SC::StochOMContext{T},
                   dacc::T,
                   alg::SOM) where {T<:Real,I<:Int}
    wmin = SC.mesh[1]
    wmax = SC.mesh[end]
    csize = length(SE.C)

    # Choose a box randomly
    t = rand(MC.rng, 1:csize)

    # Retreive the box t
    R = SE.C[t]

    # Determine left and right boundaries for the center of the box
    dx_min = wmin + R.w / T(2) - R.c
    dx_max = wmax - R.w / T(2) - R.c
    if dx_max ≤ dx_min
        return
    end

    # Calculate δc and generate shifted box
    δc = Pdx(dx_min, dx_max, MC.rng)
    if !constraints(R.c + δc - R.w/2, R.c + δc + R.w/2)
        return
    end
    Rₙ = Box(R.h, R.w, R.c + δc)

    # Calculate update for Λ
    G₁ = SE.Λ[:, t]
    G₂ = eval_lambda(Rₙ, SC.wn, SC.𝕊ᵥ)

    # Calculate new Δ function, it is actually the error function.
    Δ = calc_error(SE.G - G₁ + G₂, SC.Gᵥ, SC.σ¹)

    # Apply the Metropolis algorithm
    if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
        # Update box t
        SE.C[t] = Rₙ

        # Update Δ, G, and Λ.
        SE.Δ = Δ
        @. SE.G = SE.G - G₁ + G₂
        @. SE.Λ[:, t] = G₂

        # Update the counter
        MC.Macc[3] = MC.Macc[3] + 1
    end

    # Update the counter
    return MC.Mtry[3] = MC.Mtry[3] + 1
end

"""
    try_width(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Change the width and height of given box in the field configuration. Note
that the box's area is kept.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.
* alg  -> A StochOM struct. It contains parameters for the algorithm.

### Returns
N/A
"""
function try_width(MC::StochOMMC{I},
                   SE::StochOMElement{T},
                   SC::StochOMContext{T},
                   dacc::T,
                   alg::SOM) where {T<:Real,I<:Int}
    wbox = T(alg.wbox)
    wmin = SC.mesh[1]
    wmax = SC.mesh[end]
    csize = length(SE.C)

    # Choose a box randomly
    t = rand(MC.rng, 1:csize)

    # Retreive the box t
    R = SE.C[t]

    # Determine left and right boundaries for the width of the box
    weight = R.h * R.w
    dx_min = wbox - R.w
    dx_max = min(T(2) * (R.c - wmin), T(2) * (wmax - R.c)) - R.w
    if dx_max ≤ dx_min
        return
    end

    # Calculate δw and generate new box
    dw = Pdx(dx_min, dx_max, MC.rng)
    w = R.w + dw
    h = weight / w
    c = R.c
    if !constraints(c - w/2, c + w/2)
        return
    end
    Rₙ = Box(h, w, c)

    # Calculate update for Λ
    G₁ = SE.Λ[:, t]
    G₂ = eval_lambda(Rₙ, SC.wn, SC.𝕊ᵥ)

    # Calculate new Δ function, it is actually the error function.
    Δ = calc_error(SE.G - G₁ + G₂, SC.Gᵥ, SC.σ¹)

    # Apply the Metropolis algorithm
    if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
        # Update box t
        SE.C[t] = Rₙ

        # Update Δ, G, and Λ.
        SE.Δ = Δ
        @. SE.G = SE.G - G₁ + G₂
        @. SE.Λ[:, t] = G₂

        # Update the counter
        MC.Macc[4] = MC.Macc[4] + 1
    end

    # Update the counter
    return MC.Mtry[4] = MC.Mtry[4] + 1
end

"""
    try_height(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Change the heights of two given boxes in the field configuration.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.
* alg  -> A StochOM struct. It contains parameters for the algorithm.

### Returns
N/A
"""
function try_height(MC::StochOMMC{I},
                    SE::StochOMElement{T},
                    SC::StochOMContext{T},
                    dacc::T,
                    alg::SOM) where {T<:Real,I<:Int}
    sbox = T(alg.sbox)
    csize = length(SE.C)

    # Choose two boxes randomly
    t₁ = rand(MC.rng, 1:csize)
    t₂ = rand(MC.rng, 1:csize)
    #
    while t₁ == t₂
        t₂ = rand(MC.rng, 1:csize)
    end

    # Get box t₁ and box t₂
    R₁ = SE.C[t₁]
    R₂ = SE.C[t₂]

    # Determine left and right boundaries for the height of the box t₁
    w₁ = R₁.w
    w₂ = R₂.w
    h₁ = R₁.h
    h₂ = R₂.h
    dx_min = sbox / w₁ - h₁
    dx_max = (h₂ - sbox / w₂) * w₂ / w₁
    if dx_max ≤ dx_min
        return
    end

    # Calculate δh and generate new box t₁ and box t₂
    dh = Pdx(dx_min, dx_max, MC.rng)
    R₁ₙ = Box(R₁.h + dh, R₁.w, R₁.c)
    R₂ₙ = Box(R₂.h - dh * w₁ / w₂, R₂.w, R₂.c)

    # Calculate update for Λ
    G₁A = SE.Λ[:, t₁]
    G₁B = eval_lambda(R₁ₙ, SC.wn, SC.𝕊ᵥ)
    G₂A = SE.Λ[:, t₂]
    G₂B = eval_lambda(R₂ₙ, SC.wn, SC.𝕊ᵥ)

    # Calculate new Δ function, it is actually the error function.
    Δ = calc_error(SE.G - G₁A + G₁B - G₂A + G₂B, SC.Gᵥ, SC.σ¹)

    # Apply the Metropolis algorithm
    if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
        # Update box t₁ and box t₂
        SE.C[t₁] = R₁ₙ
        SE.C[t₂] = R₂ₙ

        # Update Δ, G, and Λ.
        SE.Δ = Δ
        @. SE.G = SE.G - G₁A + G₁B - G₂A + G₂B
        @. SE.Λ[:, t₁] = G₁B
        @. SE.Λ[:, t₂] = G₂B

        # Update the counter
        MC.Macc[5] = MC.Macc[5] + 1
    end

    # Update the counter
    return MC.Mtry[5] = MC.Mtry[5] + 1
end

"""
    try_split(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Split a given box into two boxes in the field configuration.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.
* alg  -> A StochOM struct. It contains parameters for the algorithm.

### Returns
N/A
"""
function try_split(MC::StochOMMC{I},
                   SE::StochOMElement{T},
                   SC::StochOMContext{T},
                   dacc::T,
                   alg::SOM) where {T<:Real,I<:Int}
    wbox = T(alg.wbox)
    sbox = T(alg.sbox)
    wmin = SC.mesh[1]
    wmax = SC.mesh[end]
    csize = length(SE.C)

    # Choose a box randomly
    t = rand(MC.rng, 1:csize)

    # Retreive the box t
    R₁ = SE.C[t]
    if R₁.w ≤ T(2) * wbox || R₁.w * R₁.h ≤ T(2) * sbox
        return
    end

    # Determine height for new boxes (h and h)
    h = R₁.h

    # Determine width for new boxes (w₁ and w₂)
    w₁ = wbox + (R₁.w - T(2) * wbox) * rand(MC.rng, T)
    w₂ = R₁.w - w₁
    if w₁ > w₂
        w₁, w₂ = w₂, w₁
    end

    # Determine center for new boxes (c₁ + δc₁ and c₂ + δc₂)
    c₁ = R₁.c - R₁.w / T(2) + w₁ / T(2)
    c₂ = R₁.c + R₁.w / T(2) - w₂ / T(2)
    dx_min = wmin + w₁ / T(2) - c₁
    dx_max = wmax - w₁ / T(2) - c₁
    if dx_max ≤ dx_min
        return
    end
    δc₁ = Pdx(dx_min, dx_max, MC.rng)
    δc₂ = -T(1) * w₁ * δc₁ / w₂
    if !constraints(c₁ + δc₁ - w₁/2, c₁ + δc₁ + w₁/2) ||
       !constraints(c₂ + δc₂ - w₂/2, c₂ + δc₂ + w₂/2)
        return
    end

    if (c₁ + δc₁ ≥ wmin + w₁ / T(2)) &&
       (c₁ + δc₁ ≤ wmax - w₁ / T(2)) &&
       (c₂ + δc₂ ≥ wmin + w₂ / T(2)) &&
       (c₂ + δc₂ ≤ wmax - w₂ / T(2))

        # Generate two new boxes
        R₂ = Box(h, w₁, c₁ + δc₁)
        R₃ = Box(h, w₂, c₂ + δc₂)

        # Calculate update for Λ
        G₁ = SE.Λ[:, t]
        G₂ = eval_lambda(R₂, SC.wn, SC.𝕊ᵥ)
        G₃ = eval_lambda(R₃, SC.wn, SC.𝕊ᵥ)

        # Calculate new Δ function, it is actually the error function.
        Δ = calc_error(SE.G - G₁ + G₂ + G₃, SC.Gᵥ, SC.σ¹)

        # Apply the Metropolis algorithm
        if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
            # Remove old box t and insert two new boxes
            SE.C[t] = R₂
            push!(SE.C, R₃)

            # Update Δ, G, and Λ.
            SE.Δ = Δ
            @. SE.G = SE.G - G₁ + G₂ + G₃
            @. SE.Λ[:, t] = G₂
            @. SE.Λ[:, csize+1] = G₃

            # Update the counter
            MC.Macc[6] = MC.Macc[6] + 1
        end
    end

    # Update the counter
    return MC.Mtry[6] = MC.Mtry[6] + 1
end

"""
    try_merge(
        MC::StochOMMC{I},
        SE::StochOMElement{T},
        SC::StochOMContext{T},
        dacc::T,
        alg::SOM
    ) where {T<:Real, I<:Int}

Merge two given boxes into one box in the field configuration.

### Arguments
* MC -> A StochOMMC struct. It containts some counters.
* SE -> A StochOMElement struct. It contains Monte Carlo configurations.
* SC -> A StochOMContext struct. It contains grid, mesh, and Gᵥ.
* dacc -> A predefined parameter used to calculate transition probability.
* alg  -> A StochOM struct. It contains parameters for the algorithm.

### Returns
N/A
"""
function try_merge(MC::StochOMMC{I},
                   SE::StochOMElement{T},
                   SC::StochOMContext{T},
                   dacc::T,
                   alg::SOM) where {T<:Real,I<:Int}
    wmin = SC.mesh[1]
    wmax = SC.mesh[end]
    csize = length(SE.C)

    # Choose two boxes randomly
    # Box t₂ will be removed, while box t₁ will be modified.
    t₁ = rand(MC.rng, 1:csize)
    t₂ = rand(MC.rng, 1:csize)
    #
    while t₁ == t₂
        t₂ = rand(MC.rng, 1:csize)
    end
    #
    if t₁ > t₂
        t₁, t₂ = t₂, t₁
    end

    # Get box t₁ and box t₂
    R₁ = SE.C[t₁]
    R₂ = SE.C[t₂]

    # Determine h, w, and c for new box
    weight = R₁.h * R₁.w + R₂.h * R₂.w
    w_new = T(0.5) * (R₁.w + R₂.w)
    h_new = weight / w_new
    c_new = R₁.c + (R₂.c - R₁.c) * R₂.h * R₂.w / weight

    # Determine left and right boundaries for the center of the new box
    dx_min = wmin + w_new / T(2) - c_new
    dx_max = wmax - w_new / T(2) - c_new
    if dx_max ≤ dx_min
        return
    end

    # Calculate δc and generate new box
    δc = Pdx(dx_min, dx_max, MC.rng)
    if !constraints(c_new + δc - w_new/2, c_new + δc + w_new/2)
        return
    end
    Rₙ = Box(h_new, w_new, c_new + δc)

    # Calculate update for Λ
    G₁ = SE.Λ[:, t₁]
    G₂ = SE.Λ[:, t₂]
    Gₑ = SE.Λ[:, csize]
    Gₙ = eval_lambda(Rₙ, SC.wn, SC.𝕊ᵥ)

    # Calculate new Δ function, it is actually the error function.
    Δ = calc_error(SE.G - G₁ - G₂ + Gₙ, SC.Gᵥ, SC.σ¹)

    # Apply the Metropolis algorithm
    if rand(MC.rng, T) < ((SE.Δ/Δ) ^ (T(1) + dacc))
        # Update box t₁ with new box
        SE.C[t₁] = Rₙ

        # Delete box t₂
        if t₂ < csize
            SE.C[t₂] = SE.C[end]
        end
        pop!(SE.C)

        # Update Δ, G, and Λ.
        SE.Δ = Δ
        @. SE.G = SE.G - G₁ - G₂ + Gₙ
        @. SE.Λ[:, t₁] = Gₙ
        if t₂ < csize
            @. SE.Λ[:, t₂] = Gₑ
        end

        # Update the counter
        MC.Macc[7] = MC.Macc[7] + 1
    end

    # Update the counter
    return MC.Mtry[7] = MC.Mtry[7] + 1
end

"""
    Pdx(xmin::T, xmax::T, rng::AbstractRNG) where {T<:Real}

Try to calculate the value of δξ for every elementary update according to
the probability density function. The actual meaning of δξ depends on the
elementary update.

### Arguments
* xmin -> Minimum value of δξ.
* xmax -> Maximum value of δξ
* rng -> Random number generator.

### Returns
* N -> Value of δξ.
"""
function Pdx(xmin::T, xmax::T, rng::AbstractRNG) where {T<:Real}
    xmin_abs = abs(xmin)
    xmax_abs = abs(xmax)
    X = max(xmin_abs, xmax_abs)

    γ = T(2)
    γ_X = γ / X

    η = rand(rng, T)
    𝑁 = (1 - η) * copysign(expm1(-γ_X * xmin_abs), xmin)
    𝑁 += η * copysign(expm1(-γ_X * xmax_abs), xmax)

    return copysign(log1p(-abs(𝑁)) / γ_X, 𝑁)
end
