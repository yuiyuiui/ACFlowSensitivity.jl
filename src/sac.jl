#
# Project : Gardenia
# Source  : sac.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/09/30
#

#=
### *Customized Structs* : *StochAC Solver*
=#

"""
    StochACMC

Mutable struct. It is used within the StochAC solver. It includes random
number generator and some counters.

### Members
* rng  -> Random number generator.
* Macc -> Counter for move operation (accepted).
* Mtry -> Counter for move operation (tried).
* Sacc -> Counter for swap operation (accepted).
* Stry -> Counter for swap operation (tried).

See also: [`StochACSolver`](@ref).
"""
mutable struct StochACMC{I<:Int}
    rng::AbstractRNG
    Macc::Vector{I}
    Mtry::Vector{I}
    Sacc::Vector{I}
    Stry::Vector{I}
end

"""
    StochACElement

Mutable struct. It is used to record the field configurations, which will
be sampled by Monte Carlo sweeping procedure.

### Members
* Γₚ -> It means the positions of the δ functions.
* Γₐ -> It means the weights / amplitudes of the δ functions.
"""
mutable struct StochACElement{I<:Int,T<:Real}
    Γₚ::Array{I,2}
    Γₐ::Array{T,2}
end

"""
    StochACContext

Mutable struct. It is used within the StochAC solver only.

### Members
* Gᵥ     -> Input data for correlator.
* σ¹     -> Actually 1 / σ¹.
* allow  -> Allowable indices.
* grid   -> Imaginary axis grid for input data.
* mesh   -> Real frequency mesh for output spectrum.
* mesh_weight -> Weight of the real frequency mesh.
* model  -> Default model function.
* kernel -> Default kernel function.
* Aout   -> Calculated spectral function, it is actually ⟨n(x)⟩.
* Δ      -> Precomputed δ functions.
* hτ     -> α-resolved h(τ).
* Hα     -> α-resolved Hc.
* Uα     -> α-resolved internal energy, it is actually ⟨Hα⟩.
* αₗ     -> Vector of the α parameters.
"""
mutable struct StochACContext{I<:Int,T<:Real}
    Gᵥ::Vector{T}
    σ¹::T
    allow::Vector{I}
    grid::Vector{T}
    mesh::Vector{T}
    mesh_weight::Vector{T}
    model::Vector{T}
    kernel::Array{T,2}
    Aout::Array{T,2}
    Δ::Array{T,2}
    hτ::Array{T,2}
    Hα::Vector{T}
    Uα::Vector{T}
    αₗ::Vector{T}
end

#=
### *Global Drivers*
=#

"""
    solve(S::StochACSolver, rd::RawData)

Solve the analytic continuation problem by the stochastic analytic
continuation algorithm (K. S. D. Beach's version). This is the driver for
the StochAC solver.

If the input correlators are bosonic, this solver will return A(ω) / ω
via `Asum`, instead of A(ω). At this time, `Asum` is not compatible with
`Gout`. If the input correlators are fermionic, this solver will return
A(ω) in `Asum`. Now it is compatible with `Gout`. These behaviors are just
similar to the MaxEnt, StochSK, and StochOM solvers.

Now the StochAC solver supports both continuous and δ-like spectra.

### Arguments
* S -> A StochACSolver struct.
* rd -> A RawData struct, containing raw data for input correlator.

### Returns
* mesh -> Real frequency mesh, ω.
* Asum -> Final spectral function, A(ω). Note that it is α-averaged.
* Gout -> Retarded Green's function, G(ω).
"""
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::SAC) where {T<:Real}
    fine_mesh = collect(range(ctx.mesh[1], ctx.mesh[end], alg.nfine)) # ssk needs high-precise linear grid

    # Initialize counters for Monte Carlo engine
    MC = init_mc(alg)
    println("Create infrastructure for Monte Carlo sampling")

    # Initialize Monte Carlo configurations
    SE = init_element(alg, MC.rng, T)
    println("Randomize Monte Carlo configurations")

    # Prepare some key variables
    SC = init_context(SE, GFV, fine_mesh, ctx, alg)
    println("Initialize context for the StochSK solver")

    Aout, Uα = run!(MC, SE, SC, alg)
    Asum = last!(SC, Aout, Uα)   # Average on α

    if ctx.spt isa Delta
        p = ctx.mesh[find_peaks(ctx.mesh, Asum, ctx.fp_mp; wind=ctx.fp_ww)]
        length(p) != alg.npole && @warn("Number of poles is not correct")
        γ = ones(T, alg.npole) / alg.npole
        return SC.mesh, Asum, (p, γ)
    elseif ctx.spt isa Cont
        return SC.mesh, Asum
    else
        error("Unsupported spectral function type")
    end
end

"""
    run!(MC::StochACMC, SE::StochACElement, SC::StochACContext, alg::SAC)

Perform stochastic analytic continuation simulation, sequential version.

### Arguments
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.

### Returns
* Aout -> Spectral function, A(ω).
* Uα -> α-resolved internal energy.
"""
function run!(MC::StochACMC{I}, SE::StochACElement{I,T}, SC::StochACContext{I,T},
              alg::SAC) where {I<:Int,T<:Real}
    # By default, we should write the analytic continuation results
    # into the external files.

    # Setup essential parameters
    nstep = alg.nstep
    output_per_steps = alg.ndump
    measure_per_steps = 100

    # Warmup the Monte Carlo engine
    println("Start thermalization...")
    warmup!(MC, SE, SC, alg)

    # Sample and collect data
    step = T(0)
    println("Start stochastic sampling...")
    for iter in 1:nstep
        sample!(MC, SE, SC, alg)

        if iter % measure_per_steps == 0
            step = step + T(1)
            measure!(SE, SC, alg)
        end

        if iter % output_per_steps == 0
            prog = round(I, iter / nstep * 100)
            @show iter
            @show prog
        end
    end

    return average(step, SC)
end

"""
    average(step::T, SC::StochACContext{I,T}) where {I<:Int,T<:Real}

Postprocess the results generated during the stochastic analytic
continuation simulations. It will calculate the spectral functions, and
α-resolved internal energies.

### Arguments
* step -> How many steps are there in the Monte Carlo samplings.
* SC   -> A StochACContext struct.

### Returns
* Aout -> Spectral function, A(ω,α).
* Uα -> α-resolved internal energy.
"""
function average(step::T, SC::StochACContext{I,T}) where {I<:Int,T<:Real}
    # Get key parameters
    nmesh = length(SC.mesh)
    nalph = length(SC.αₗ)

    # Renormalize the spectral functions
    Aout = zeros(T, nmesh, nalph)
    for i in 1:nalph
        for j in 1:nmesh
            Aout[j, i] = SC.Aout[j, i] * SC.model[j] / T(π) / step
        end
    end

    # Renormalize the internal energies
    Uα = SC.Uα / step

    return Aout, Uα
end

"""
    last!(SC::StochACContext{I,T}, Aout::Array{T,2}, Uα::Vector{T}) where {I<:Int,T<:Real}

It will process and write the calculated results by the StochAC solver,
including effective hamiltonian, final spectral function, reproduced
correlator.

### Arguments
* SC   -> A StochACContext struct.
* Aout -> α-dependent spectral functions.
* Uα   -> α-dependent internal energies.

### Returns
* Asum -> Final spectral function (α-averaged), A(ω).
* G -> Retarded Green's function, G(ω).
"""
function last!(SC::StochACContext{I,T}, Aout::Array{T,2},
               Uα::Vector{T}) where {I<:Int,T<:Real}
    function fitfun(x, p)
        return @. p[1] * x + p[2]
    end

    # Get dimensional parameters
    nmesh, nalph = size(Aout)

    # Try to fit the internal energies to find out optimal α
    guess = [T(1), T(1)]
    fit_l = curve_fit(fitfun, SC.αₗ[1:5], log10.(Uα[1:5]), guess)
    fit_r = curve_fit(fitfun, SC.αₗ[(end - 4):end], log10.(Uα[(end - 4):end]), guess)
    a, b = fit_l.param
    c, d = fit_r.param
    aopt = (d - b) / (a - c)
    close = argmin(abs.(SC.αₗ .- aopt))
    println("Fitting parameters [a,b] are: [ $a, $b ]")
    println("Fitting parameters [c,d] are: [ $c, $d ]")
    println("Perhaps the optimal α is: ", aopt)

    # Calculate final spectral functions and write them
    Asum = zeros(T, nmesh)
    for i in close:(nalph - 1)
        @. Asum = Asum + (Uα[i] - Uα[i+1]) * Aout[:, i]
    end
    @. Asum = Asum / (Uα[close] - Uα[end])

    return Asum
end

#=
### *Core Algorithms*
=#

"""
    warmup!(MC::StochACMC, SE::StochACElement, SC::StochACContext, alg::SAC)

Warmup the Monte Carlo engine to acheieve thermalized equilibrium. After
that, the Monte Carlo counters will be reset.

### Arguments
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.

### Returns
N/A
"""
function warmup!(MC::StochACMC{I}, SE::StochACElement{I,T}, SC::StochACContext{I,T},
                 alg::SAC) where {I<:Int,T<:Real}
    # Set essential parameter
    nwarm = alg.nwarm

    # Shuffle the Monte Carlo field configuration
    for iter in 1:nwarm
        sample!(MC, SE, SC, alg)
    end

    # Reset the counters
    fill!(MC.Macc, T(0))
    fill!(MC.Mtry, T(0))

    fill!(MC.Sacc, T(0))
    return fill!(MC.Stry, T(0))
end

"""
    sample!(MC::StochACMC, SE::StochACElement, SC::StochACContext, alg::SAC)

Perform Monte Carlo sweeps and sample the field configurations.

### Arguments
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.

### Returns
N/A
"""
function sample!(MC::StochACMC{I}, SE::StochACElement{I,T}, SC::StochACContext{I,T},
                 alg::SAC) where {I<:Int,T<:Real}
    nalph = alg.nalph

    if rand(MC.rng) < 0.9
        if rand(MC.rng) > 0.5
            for i in 1:nalph
                try_move_a!(i, MC, SE, SC, alg)
            end
        else
            if rand(MC.rng) > 0.2
                for i in 1:nalph
                    try_move_s!(i, MC, SE, SC, alg)
                end
            else
                for i in 1:nalph
                    try_move_p!(i, MC, SE, SC, alg)
                end
            end
        end
    else
        if nalph > 1
            try_move_x!(MC, SE, SC, alg)
        end
    end
end

"""
    measure!(SE::StochACElement, SC::StochACContext, alg::SAC)

Accumulate the α-resolved spectral functions and internal energies.

### Arguments
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.

### Returns
N/A
"""
function measure!(SE::StochACElement{I,T}, SC::StochACContext{I,T},
                  alg::SAC) where {I<:Int,T<:Real}
    nalph = alg.nalph

    # Loop over each α parameter
    for ia in 1:nalph
        da = view(SE.Γₐ, :, ia)
        dp = view(SE.Γₚ, :, ia)
        SC.Aout[:, ia] = SC.Aout[:, ia] .+ SC.Δ[:, dp] * da
        SC.Uα[ia] = SC.Uα[ia] + SC.Hα[ia]
    end
end

#=
### *Service Functions*
=#

"""
    init_mc(alg::SAC)

Try to create a StochACMC struct. Some counters for Monte Carlo updates
are initialized here.

### Arguments
* alg -> A SAC struct.

### Returns
* MC -> A StochACMC struct.

See also: [`StochACMC`](@ref).
"""
function init_mc(alg::SAC)
    nalph = alg.nalph
    #
    seed = rand(1:100000000)
    rng = MersenneTwister(seed)
    #
    Macc = zeros(Int64, nalph)
    Mtry = zeros(Int64, nalph)
    Sacc = zeros(Int64, nalph)
    Stry = zeros(Int64, nalph)
    #
    MC = StochACMC(rng, Macc, Mtry, Sacc, Stry)

    return MC
end

"""
    init_element(
        alg::SAC,
        rng::AbstractRNG,
        T::Type{<:Real}
    )

Randomize the configurations for future Monte Carlo sampling. It will
return a StochACElement struct.

### Arguments
* alg -> A SAC struct.
* rng -> Random number generator.

### Returns
* SE -> A StochACElement struct.

See also: [`StochACElement`](@ref).
"""
function init_element(alg::SAC,
                      rng::AbstractRNG,
                      T::Type{<:Real})
    nalph = alg.nalph
    pn = alg.npole

    Γₚ = rand(rng, collect(1:alg.nfine), (pn, nalph))
    Γₐ = rand(rng, T, (pn, nalph))

    for j in 1:nalph
        Γⱼ = view(Γₐ, :, j)
        s = sum(Γⱼ)
        @. Γⱼ = Γⱼ / s
    end

    SE = StochACElement(Γₚ, Γₐ)

    return SE
end

"""
    init_context(
        SE::StochACElement,
        GFV::Vector{Complex{T}},
        fine_mesh::Vector{T},
        ctx::CtxData{T},
        alg::SAC
    )

Try to create a StochACContext struct, which contains some key variables,
including grid, mesh, input correlator and the corresponding standard
deviation, kernel matrix, spectral function, and α-resolved Hamiltonian.

### Arguments
* SE -> A StochACElement struct.
* GFV -> Input correlator. It will be changed in this function.
* fine_mesh -> Very fine mesh in [wmin, wmax].
* ctx -> Context data containing mesh and other parameters.
* alg -> SAC algorithm parameters.

### Returns
* SC -> A StochACContext struct.
"""
function init_context(SE::StochACElement,
                      GFV::Vector{Complex{T}},
                      fine_mesh::Vector{T},
                      ctx::CtxData{T},
                      alg::SAC) where {T<:Real}
    # Get parameters
    nmesh = length(ctx.mesh)
    nalph = alg.nalph

    # Allocate memory for spectral function, A(ω,α)
    Aout = zeros(T, nmesh, nalph)

    # Prepare some key variables
    # Only flat model is valid for the StochAC solver.
    model = make_model("flat", ctx)

    # Precompute δ functions
    ϕ = cumsum(model .* ctx.mesh_weight)
    Δ = calc_delta(fine_mesh, ϕ)

    # Build kernel matrix
    _, _, _, U, S, V = SingularSpace(GFV, ctx.iwn*ctx.σ, fine_mesh*ctx.σ)

    # Get new kernel matrix
    kernel = Diagonal(S) * V'

    # Get new (input) correlator
    Gᵥ = U' * (vcat(real(GFV), imag(GFV)) .* 1 / ctx.σ)

    # Precompute hamiltonian
    hτ, Hα, Uα = calc_hamil(alg.nalph, SE.Γₚ, SE.Γₐ, kernel, Gᵥ)

    # Precompute α parameters
    αₗ = calc_alpha(alg, T)

    return StochACContext(Gᵥ, 1/ctx.σ, collect(1:alg.nfine), ctx.wn, ctx.mesh,
                          ctx.mesh_weight, model,
                          kernel, Aout, Δ, hτ, Hα, Uα, αₗ)
end

"""
    calc_delta(fine_mesh::Vector{T}, ϕ::Vector{T})

Precompute the Δ functions. `fine_mesh` is a very dense mesh in [wmin, wmax]
and `ϕ` is the ϕ function.

Here we just use f(x) = η / (x² + η²) to approximate the δ function, where
η is a small parameter.

### Arguments
See above explanations.

### Returns
* Δ -> The Δ(ω) function.

See also: [`calc_phi`](@ref).
"""
function calc_delta(fine_mesh::Vector{T}, ϕ::Vector{T}) where {T<:Real}
    nmesh = length(ϕ)
    #
    nfine = length(fine_mesh)
    wmax = fine_mesh[end]
    wmin = fine_mesh[1]
    #
    η₁ = T(0.001)
    η₂ = T(0.001) ^ 2

    Δ = zeros(T, nmesh, nfine)
    s = similar(ϕ)
    for i in 1:nfine
        # We should convert the mesh `fmesh` from [wmin,wmax] to [0,1].
        𝑥 = (fine_mesh[i] - wmin) / (wmax - wmin)
        @. s = (ϕ - 𝑥) ^ 2 + η₂
        @. Δ[:, i] = η₁ / s
    end

    return Δ
end

"""
    calc_hamil(
        nalph::I,
        Γₚ::Array{I,2},
        Γₐ::Array{I,2},
        kernel::Matrix{T},
        Gᵥ::Vector{T}
    ) where {I<:Int,T<:Real}

Initialize h(τ) and H(α) using Eq.(35) and Eq.(36), respectively. `Γₚ`
and `Γₐ` represent n(x), `kernel` means the kernel function, `Gᵥ` is the
correlator. Note that `kernel` and `Gᵥ` have been rotated into singular
space. Please see comments in `init()` for more details.

### Arguments
See above explanations.

### Returns
* hτ -> α-resolved h(τ).
* Hα -> α-resolved Hc.
* Uα -> α-resolved internal energy, it is actually ⟨Hα⟩.

See also: [`calc_htau`](@ref).
"""
function calc_hamil(nalph::I,
                    Γₚ::Array{I,2},
                    Γₐ::Array{T,2},
                    kernel::Matrix{T},
                    Gᵥ::Vector{T}) where {I<:Int,T<:Real}
    ngrid = length(Gᵥ)

    hτ = zeros(T, ngrid, nalph)
    Hα = zeros(T, nalph)
    Uα = zeros(T, nalph)

    for i in 1:nalph
        hτ[:, i] = calc_htau(Γₚ[:, i], Γₐ[:, i], kernel, Gᵥ)
        Hα[i] = dot(hτ[:, i], hτ[:, i])
    end

    return hτ, Hα, Uα
end

"""
    calc_htau(
        Γₚ::Vector{I},
        Γₐ::Vector{T},
        kernel::Matrix{T},
        Gᵥ::Vector{T}
    )

Try to calculate α-dependent h(τ) via Eq.(36). `Γₚ` and `Γₐ` represent
n(x), `kernel` means the kernel function, `Gᵥ` is the correlator. Note
that `kernel` and `Gᵥ` have been rotated into singular space. Please
see comments in `init_context()` for more details.

### Arguments
See above explanations.

### Returns
* hτ -> α-resolved h(τ).

See also: [`calc_hamil`](@ref).
"""
function calc_htau(Γₚ::Vector{I}, Γₐ::Vector{T},
                   kernel::Matrix{T},
                   Gᵥ::Vector{T}) where {I<:Int,T<:Real}
    hτ = similar(Gᵥ)
    #
    for i in eachindex(Gᵥ)
        hτ[i] = dot(Γₐ, view(kernel, i, Γₚ)) - Gᵥ[i]
    end
    #
    return hτ
end

"""
    calc_alpha(alg::SAC)

Generate a list for the α parameters.

### Arguments
N/A

### Returns
* αₗ -> List of the α parameters.
"""
function calc_alpha(alg::SAC, T::Type{<:Real})
    nalph = alg.nalph
    alpha = alg.alpha
    ratio = alg.ratio

    αₗ = collect(T(alpha) * (T(ratio) ^ (x - 1)) for x in 1:nalph)

    return αₗ
end

"""
    try_move_s!(
        i::I,
        MC::StochACMC{I},
        SE::StochACElement{I,T},
        SC::StochACContext{I,T},
        alg::SAC
    ) where {I<:Int,T<:Real}

Select one δ function randomly and then change its position.

### Arguments
* i -> Index for α parameters.
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.
### Returns
N/A

See also: [`try_move_p!`](@ref).
"""
function try_move_s!(i::I,
                     MC::StochACMC{I},
                     SE::StochACElement{I,T},
                     SC::StochACContext{I,T},
                     alg::SAC) where {I<:Int,T<:Real}
    # Get current number of δ functions
    pn = alg.npole

    # Choose one δ function
    γ = rand(MC.rng, 1:pn)

    # Extract weight for the δ function
    a = SE.Γₐ[γ, i]

    # Choose new position for the δ function
    p = rand(MC.rng, SC.allow)

    # Try to calculate the change of Hc using Eq.~(42).
    hc = view(SC.hτ, :, i)
    Kₚ = view(SC.kernel, :, p)
    Kᵧ = view(SC.kernel, :, SE.Γₚ[γ, i])
    #
    δhc = a * (Kₚ - Kᵧ)
    δH = dot(δhc, T(2) * hc + δhc)

    # Apply Metropolis algorithm
    MC.Mtry[i] = MC.Mtry[i] + 1
    if δH ≤ 0.0 || exp(-SC.αₗ[i] * δH) > rand(MC.rng)
        # Update Monte Carlo configurations
        SE.Γₚ[γ, i] = p

        # Update h(τ)
        @. hc = hc + δhc

        # Update Hc
        SC.Hα[i] = SC.Hα[i] + δH

        # Update Monte Carlo counter
        MC.Macc[i] = MC.Macc[i] + 1
    end
end

"""
    try_move_p!(
        i::I,
        MC::StochACMC{I},
        SE::StochACElement{I,T},
        SC::StochACContext{I,T},
        alg::SAC
    ) where {I<:Int,T<:Real}

Select two δ functions randomly and then change their positions.

### Arguments
* i -> Index for α parameters.
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.
### Returns
N/A

See also: [`try_move_s!`](@ref).
"""
function try_move_p!(i::I,
                     MC::StochACMC{I},
                     SE::StochACElement{I,T},
                     SC::StochACContext{I,T},
                     alg::SAC) where {I<:Int,T<:Real}
    # Get current number of δ functions
    pn = alg.npole
    #
    if pn < 2
        return
    end

    # Choose two δ functions, they are labelled as γ₁ and γ₂, respectively.
    γ₁ = 1
    γ₂ = 1
    while γ₁ == γ₂
        γ₁ = rand(MC.rng, 1:pn)
        γ₂ = rand(MC.rng, 1:pn)
    end

    # Extract weights for the two δ functions (a₁ and a₂)
    a₁ = SE.Γₐ[γ₁, i]
    a₂ = SE.Γₐ[γ₂, i]

    # Choose new positions for the two δ functions (p₁ and p₂).
    # Note that their old positions are SE.Γₚ[γ₁,i] and SE.Γₚ[γ₂,i].
    p₁ = rand(MC.rng, SC.allow)
    p₂ = rand(MC.rng, SC.allow)

    # Try to calculate the change of Hc using Eq.~(42).
    hc = view(SC.hτ, :, i)
    K₁ = view(SC.kernel, :, p₁)
    K₂ = view(SC.kernel, :, p₂)
    K₃ = view(SC.kernel, :, SE.Γₚ[γ₁, i])
    K₄ = view(SC.kernel, :, SE.Γₚ[γ₂, i])
    #
    δhc = a₁ * (K₁ - K₃) + a₂ * (K₂ - K₄)
    δH = dot(δhc, T(2) * hc + δhc)

    # Apply Metropolis algorithm
    MC.Mtry[i] = MC.Mtry[i] + 1
    if δH ≤ 0.0 || exp(-SC.αₗ[i] * δH) > rand(MC.rng)
        # Update Monte Carlo configurations
        SE.Γₚ[γ₁, i] = p₁
        SE.Γₚ[γ₂, i] = p₂

        # Update h(τ)
        @. hc = hc + δhc

        # Update Hc
        SC.Hα[i] = SC.Hα[i] + δH

        # Update Monte Carlo counter
        MC.Macc[i] = MC.Macc[i] + 1
    end
end

"""
    try_move_a!(
        i::I,
        MC::StochACMC{I},
        SE::StochACElement{I,T},
        SC::StochACContext{I,T},
        alg::SAC
    ) where {I<:Int,T<:Real}

Select two δ functions randomly and then change their weights.

### Arguments
* i -> Index for α parameters.
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.
### Returns
N/A

See also: [`try_move_x!`](@ref).
"""
function try_move_a!(i::I,
                     MC::StochACMC{I},
                     SE::StochACElement{I,T},
                     SC::StochACContext{I,T},
                     alg::SAC) where {I<:Int,T<:Real}
    # Get current number of δ functions
    pn = alg.npole
    #
    if pn < 2
        return
    end

    # Choose two δ functions, they are labelled as γ₁ and γ₂, respectively.
    γ₁ = 1
    γ₂ = 1
    while γ₁ == γ₂
        γ₁ = rand(MC.rng, 1:pn)
        γ₂ = rand(MC.rng, 1:pn)
    end

    # Extract weights for the two δ functions (a₃ and a₄), then try to
    # calculate new weights for them (a₁ and a₂).
    a₁ = T(0)
    a₂ = T(0)
    a₃ = SE.Γₐ[γ₁, i]
    a₄ = SE.Γₐ[γ₂, i]
    δa = T(0)
    while true
        δa = rand(MC.rng) * (a₃ + a₄) - a₃
        a₁ = a₃ + δa
        a₂ = a₄ - δa
        if a₁ > 0 && a₂ > 0
            break
        end
    end

    # Try to calculate the change of Hc using Eq.~(42).
    hc = view(SC.hτ, :, i)
    K₁ = view(SC.kernel, :, SE.Γₚ[γ₁, i])
    K₂ = view(SC.kernel, :, SE.Γₚ[γ₂, i])
    #
    δhc = δa * (K₁ - K₂)
    δH = dot(δhc, T(2) * hc + δhc)

    # Apply Metropolis algorithm
    MC.Mtry[i] = MC.Mtry[i] + 1
    if δH ≤ 0.0 || exp(-SC.αₗ[i] * δH) > rand(MC.rng)
        # Update Monte Carlo configurations
        SE.Γₐ[γ₁, i] = a₁
        SE.Γₐ[γ₂, i] = a₂

        # Update h(τ)
        @. hc = hc + δhc

        # Update Hc
        SC.Hα[i] = SC.Hα[i] + δH

        # Update Monte Carlo counter
        MC.Macc[i] = MC.Macc[i] + 1
    end
end

"""
    try_move_x!(
        MC::StochACMC{I},
        SE::StochACElement{I,T},
        SC::StochACContext{I,T},
        alg::SAC
    ) where {I<:Int,T<:Real}

Try to exchange field configurations between two adjacent layers. Because
this function involves two layers, so it doesn't need the argument `i`.

### Arguments
* MC -> A StochACMC struct.
* SE -> A StochACElement struct.
* SC -> A StochACContext struct.
* alg -> A SAC struct.
### Returns
N/A

See also: [`try_move_a!`](@ref).
"""
function try_move_x!(MC::StochACMC{I},
                     SE::StochACElement{I,T},
                     SC::StochACContext{I,T},
                     alg::SAC) where {I<:Int,T<:Real}
    # Get number of α parameters
    nalph = alg.nalph

    # Select two adjacent layers (two adjacent α parameters)
    i = rand(MC.rng, 1:nalph)
    j = rand(MC.rng) > 0.5 ? i + 1 : i - 1
    i == 1 && (j = i + 1)
    i == nalph && (j = i - 1)

    # Calculate change of Hc
    δα = SC.αₗ[i] - SC.αₗ[j]
    δH = SC.Hα[i] - SC.Hα[j]

    # Apply Metropolis algorithm
    MC.Stry[i] = MC.Stry[i] + 1
    MC.Stry[j] = MC.Stry[j] + 1
    if exp(δα * δH) > rand(MC.rng)
        # Update Monte Carlo configurations
        SE.Γₚ[:, i], SE.Γₚ[:, j] = SE.Γₚ[:, j], SE.Γₚ[:, i]
        SE.Γₐ[:, i], SE.Γₐ[:, j] = SE.Γₐ[:, j], SE.Γₐ[:, i]

        # Update h(τ) and Hc
        SC.hτ[:, i], SC.hτ[:, j] = SC.hτ[:, j], SC.hτ[:, i]
        SC.Hα[i], SC.Hα[j] = SC.Hα[j], SC.Hα[i]

        # Update Monte Carlo counters
        MC.Sacc[i] = MC.Sacc[i] + 1
        MC.Sacc[j] = MC.Sacc[j] + 1
    end
end
