"""
	StochSKElement

Mutable struct. It is used to record the field configurations, which will
be sampled by Monte Carlo sweeping procedure.

In the present implementation of StochSK solver, the amplitudes of the δ
functions are fixed. But in principles, they could be sampled in the Monte
Carlo procedure.

### Members
* P -> It means the positions of the δ functions.
* A -> It means the weights / amplitudes of the δ functions.
* W -> It denotes the window that is used to shift the δ functions.
"""
mutable struct StochSKElement{I<:Int,T<:Real}
    P::Vector{I}
    A::T
    W::I
end

"""
	StochSKContext

Mutable struct. It is used within the StochSK solver only.

### Members
* Gᵥ     -> Input data for correlator.
* Gᵧ     -> Generated correlator.
* σinv   -> Actually 1 / σ.
* allow  -> Allowable indices.
* grid   -> Imaginary axis grid for input data.
* mesh   -> Real frequency mesh for output spectrum.
* kernel -> Default kernel function.
* Aout   -> Calculated spectral function.
* χ²     -> Current goodness-of-fit function.
* χ²min  -> Mininum goodness-of-fit function.
* χ²vec  -> Vector of goodness-of-fit function.
* Θ      -> Current Θ parameter.
* Θvec   -> Vector of Θ parameter.
"""
mutable struct StochSKContext{I<:Int,T<:Real}
    Gᵥ::Vector{T}
    Gᵧ::Vector{T}
    σinv::T
    allow::Vector{I}
    grid::Vector{T}
    mesh::Vector{T}
    mesh_weight::Vector{T}
    kernel::Array{T,2}
    Aout::Vector{T}
    χ²::T
    χ²min::T
    χ²vec::Vector{T}
    Θ::T
    Θvec::Vector{T}
end

"""
	StochSKMC

Mutable struct. It is used within the StochSK solver. It includes random
number generator and some counters.

### Members
* rng  -> Random number generator.
* Sacc -> Counter for single-updated operation (accepted).
* Stry -> Counter for single-updated operation (tried).
* Pacc -> Counter for pair-updated operation (accepted).
* Ptry -> Counter for pair-updated operation (tried).
* Qacc -> Counter for quadruple-updated operation (accepted).
* Qtry -> Counter for quadruple-updated operation (tried).

See also: [`StochSKSolver`](@ref).
"""
mutable struct StochSKMC{I<:Int}
    rng::AbstractRNG
    Sacc::I
    Stry::I
    Pacc::I
    Ptry::I
    Qacc::I
    Qtry::I
end

#=
### *Global Drivers*
=#

"""
	solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::SSK) where {T<:Real}

Main driver function for the StochSK solver.

### Arguments
* GFV -> Input Green's function data.
* ctx -> Context data containing mesh and other parameters.
* alg -> SSK algorithm parameters.

### Returns
* mesh -> Real frequency mesh.
* Aout -> Spectral function.
"""
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::SSK) where {T<:Real}
    println("[ StochSK ]")
    fine_mesh = collect(range(ctx.mesh[1], ctx.mesh[end], alg.nfine)) # ssk needs high-precise linear grid

    # Initialize counters for Monte Carlo engine
    MC = init_mc(alg)
    println("Create infrastructure for Monte Carlo sampling")

    # Initialize Monte Carlo configurations
    SE = init_element(alg, MC.rng, ctx)
    println("Randomize Monte Carlo configurations")

    # Prepare some key variables
    SC = init_context(SE, GFV, fine_mesh, ctx, alg)
    println("Initialize context for the StochSK solver")

    Aout, _, _ = run!(MC, SE, SC, alg)
    if ctx.spt isa Delta
        p = ctx.mesh[find_peaks(ctx.mesh, Aout, ctx.fp_mp; wind=ctx.fp_ww)]
        # If length(p) != npole, then we just use SE.P
        if length(p) != alg.npole
            p = T[]
            for p_fine in SE.P
                idx = nearest(SC.mesh, p_fine / alg.nfine)
                push!(p, SC.mesh[idx])
            end
            sort!(p)
        end
        # γ = poles2realγ(p, GFV, ctx.iwn)
        γ = ones(T, alg.npole) / alg.npole

        return SC.mesh, Aout, (p, γ)
    elseif ctx.spt isa Cont
        return SC.mesh, Aout
    else
        error("Unsupported spectral function type")
    end
end

"""
	run!(MC::StochSKMC, SE::StochSKElement, SC::StochSKContext, alg::SSK)

Perform stochastic analytic continuation simulation, sequential version.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
* Aout -> Spectral function, A(ω).
* χ²vec -> Θ-dependent χ², χ²(Θ).
* Θvec -> List of Θ parameters.
"""
function run!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
              alg::SSK) where {I<:Int,T<:Real}

    # Setup essential parameters
    nstep = alg.nstep
    retry = alg.retry
    measure_per_steps = 10

    # Warmup the Monte Carlo engine
    println("Start thermalization...")
    warmup(MC, SE, SC, alg)

    # Shuffle the Monte Carlo configuration again
    shuffle(MC, SE, SC, alg)

    # Sample and collect data
    step = T(0)
    println("Start stochastic sampling...")
    for iter in 1:nstep
        if iter % retry == 0
            SC.χ² = calc_goodness(SC.Gᵧ, SC.Gᵥ)
        end

        sample!(MC, SE, SC, alg)

        if iter % measure_per_steps == 0
            step = step + T(1)
            measure!(SE, SC, alg)
        end
    end

    return average(step, SC)
end

"""
	average(step::T, SC::StochSKContext{I,T}) where {I<:Int,T<:Real}

Postprocess the results generated during the stochastic analytic
continuation simulations. It will generate the spectral functions.

### Arguments
* step -> Number of Monte Carlo samplings.
* SC   -> A StochSKContext struct.

### Returns
* Aout -> Spectral function, A(ω).
* χ²vec -> Θ-dependent χ², χ²(Θ).
* Θvec -> List of Θ parameters.
"""
function average(step::T, SC::StochSKContext{I,T}) where {I<:Int,T<:Real}
    SC.Aout = SC.Aout ./ (step * SC.mesh_weight)
    return SC.Aout, SC.χ²vec, SC.Θvec
end

#=
### *Core Algorithms*
=#

"""
	warmup(MC::StochSKMC, SE::StochSKElement, SC::StochSKContext, alg::SSK)

Warmup the Monte Carlo engine to acheieve thermalized equilibrium. Then
it will try to figure out the optimized Θ and the corresponding Monte
Carlo field configuration.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
N/A
"""
function warmup(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                alg::SSK) where {I<:Int,T<:Real}
    # Get essential parameters
    nwarm = alg.nwarm
    ratio = T(alg.ratio)
    threshold = T(1e-3)

    # To store the historic Monte Carlo field configurations
    𝒞ᵧ = StochSKElement[]

    # Change the Θ parameter and approch the equilibrium state
    for i in 1:nwarm
        # Shuffle the Monte Carlo configurations
        shuffle(MC, SE, SC, alg)

        # Backup key parameters and Monte Carlo field configurations
        SC.χ²vec[i] = SC.χ²
        SC.Θvec[i] = SC.Θ
        push!(𝒞ᵧ, deepcopy(SE))

        # Check whether the equilibrium state is reached
        δχ² = SC.χ² - SC.χ²min
        # @printf("step : %5i ", i)
        # @printf("χ² - χ²min -> %12.6e\n", δχ²)
        if δχ² < threshold
            println("Reach equilibrium state")
            break
        else
            if i == nwarm
                error("Fail to reach equilibrium state")
            end
        end

        # Adjust the Θ parameter
        SC.Θ = SC.Θ * ratio
    end

    # Well, we have vectors for Θ and χ². We have to figure out the
    # optimized Θ and χ², and then extract the corresponding Monte
    # Carlo field configuration.
    c = calc_theta(length(𝒞ᵧ), SC, alg)
    @assert 1 ≤ c ≤ length(𝒞ᵧ)

    # Retrieve the Monte Carlo field configuration
    @. SE.P = 𝒞ᵧ[c].P
    SE.A = 𝒞ᵧ[c].A
    SE.W = 𝒞ᵧ[c].W

    # Reset Θ
    SC.Θ = SC.Θvec[c]

    # Update Gᵧ and χ²
    SC.Gᵧ = calc_correlator(SE, SC.kernel)
    SC.χ² = calc_goodness(SC.Gᵧ, SC.Gᵥ)
    return println("Θ = ", SC.Θ, " χ² = ", SC.χ², " (step = $c)")
end

"""
	sample!(MC::StochSKMC, SE::StochSKElement, SC::StochSKContext)

Perform Monte Carlo sweeps and sample the field configurations.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.

### Returns
N/A
"""
function sample!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                 alg::SSK) where {I<:Int,T<:Real}
    if rand(MC.rng) < 0.80
        try_move_s!(MC, SE, SC, alg)
    else
        if rand(MC.rng) < 0.50
            try_move_p!(MC, SE, SC, alg)
        else
            try_move_q!(MC, SE, SC, alg)
        end
    end
end

"""
	measure!(SE::StochSKElement, SC::StochSKContext, alg::SSK)

Accumulate the final spectral functions A(ω).

### Arguments
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
N/A

See also: [`nearest`](@ref).
"""
function measure!(SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                  alg::SSK) where {I<:Int,T<:Real}
    nfine = alg.nfine
    pn = alg.npole

    for j in 1:pn
        d_pos = SE.P[j]
        # d_pos / nfine denotes the position of the selected δ-like peak
        # in the fine linear mesh.
        #
        # The nearest() function is used to extract the approximated
        # position (index) of the selected δ function in the spectral
        # mesh, which could be linear or non-linear.
        #
        # Note that nearest() is defined in mesh.jl.
        s_pos = nearest(SC.mesh, d_pos / nfine)
        SC.Aout[s_pos] = SC.Aout[s_pos] + SE.A
    end
end

"""
	shuffle(MC::StochSKMC, SE::StochSKElement, SC::StochSKContext, alg::SSK)

Try to shuffle the Monte Carlo field configuration via the Metropolis
algorithm. Then the window for shifting the δ functions is adjusted.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
N/A
"""
function shuffle(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                 alg::SSK) where {I<:Int,T<:Real}
    # Get/set essential parameters
    nfine = alg.nfine
    retry = alg.retry
    max_bin_size = 100 # You can increase it to improve the accuracy

    # Announce counters
    bin_χ² = T(0)
    bin_acc = T(0)
    bin_try = T(0)

    # Perform Monte Carlo sweeping
    for s in 1:max_bin_size
        # Recalculate the goodness-of-fit function
        if s % retry == 0
            SC.χ² = calc_goodness(SC.Gᵧ, SC.Gᵥ)
        end

        sample!(MC, SE, SC, alg)

        # Update the counters
        bin_χ² = bin_χ² + SC.χ²
        bin_acc = bin_acc + (MC.Sacc + MC.Pacc)
        bin_try = bin_try + (MC.Stry + MC.Ptry)
    end

    # Calculate the transition probability, and then adjust the window,
    # which restricts the movement of the δ functions.
    #
    # The transition probability will be kept around 0.5.
    𝑝 = bin_acc / bin_try
    #
    if 𝑝 > 1 // 2
        r = SE.W * (3 // 2)
        if ceil(I, r) < nfine
            SE.W = ceil(I, r)
        else
            SE.W = nfine
        end
    end
    #
    if 𝑝 < 2 // 5
        SE.W = ceil(I, SE.W / (3 // 2))
    end

    # Update χ² with averaged χ²
    return SC.χ² = bin_χ² / max_bin_size
end

#=
### *Service Functions*
=#

"""
	init_mc(S::SSK)

Try to create a StochSKMC struct. Some counters for Monte Carlo updates
are initialized here.

### Arguments
* S -> A StochSKSolver struct.

### Returns
* MC -> A StochSKMC struct.

See also: [`StochSKMC`](@ref).
"""
function init_mc(alg::SSK)
    seed = rand(1:100000000)
    rng = MersenneTwister(seed)
    #
    Sacc = 0
    Stry = 0
    Pacc = 0
    Ptry = 0
    Qacc = 0
    Qtry = 0
    #
    MC = StochSKMC(rng, Sacc, Stry, Pacc, Ptry, Qacc, Qtry)

    return MC
end

"""
	init_element(
		alg::SSK,
		rng::AbstractRNG,
		ctx::CtxData{T}
	)

Randomize the configurations for future Monte Carlo sampling. It will
return a StochSKElement struct.

### Arguments
* alg   -> A SSK struct.
* rng   -> Random number generator.
* allow -> Allowed positions for the δ peaks.

### Returns
* SE -> A StochSKElement struct.

See also: [`StochSKElement`](@ref).
"""
function init_element(alg::SSK,
                      rng::AbstractRNG,
                      ctx::CtxData{T}) where {T<:Real}
    β = ctx.β
    wmax = ctx.mesh[end]
    wmin = ctx.mesh[1]
    nfine = alg.nfine
    pn = alg.npole

    position = rand(rng, 1:nfine, pn)
    #
    amplitude = T(1) / pn
    #
    δf = (wmax - wmin) / (nfine - 1)
    average_freq = abs(log(T(2)) / β)
    window_width = ceil(Int, T(0.1) * average_freq / δf)

    return StochSKElement(position, amplitude, window_width)
end

function init_context(SE::StochSKElement{I,T},
                      GFV::Vector{Complex{T}},
                      fine_mesh::Vector{T},
                      ctx::CtxData{T},
                      alg::SSK) where {I<:Int,T<:Real}

    # Get parameters
    nmesh = length(ctx.mesh)
    nwarm = alg.nwarm
    θ = T(alg.θ)

    # Allocate memory for spectral function, A(ω)
    Aout = zeros(T, nmesh)

    # Allocate memory for χ² and Θ
    χ²vec = zeros(T, nwarm)
    θvec = zeros(T, nwarm)

    # Build kernel matrix
    _, _, _, U, S, V = SingularSpace(GFV, ctx.iwn * ctx.σ, fine_mesh * ctx.σ)

    # Get new kernel matrix
    kernel = Diagonal(S) * V'

    # Get new (input) correlator
    Gᵥ = U' * (vcat(real(GFV), imag(GFV)) .* 1 / ctx.σ)

    # Calculate reconstructed correlator using current field configuration
    Gᵧ = calc_correlator(SE, kernel)

    # Calculate goodness-of-fit functional χ²
    𝚾 = calc_goodness(Gᵧ, Gᵥ)
    χ², χ²min = 𝚾, 𝚾

    return StochSKContext(Gᵥ, Gᵧ, 1 / ctx.σ, collect(1:(alg.nfine)), ctx.wn, ctx.mesh,
                          ctx.mesh_weight, kernel, Aout,
                          χ², χ²min, χ²vec, θ, θvec)
end

"""
	calc_correlator(SE::StochSKElement, kernel::Array{F64,2})

Try to calculate correlator with the kernel function and the Monte Carlo
field configuration. This correlator will then be used to evaluate the
goodness-of-fit function χ².

### Arguments
* SE     -> A StochSKElement struct.
* kernel -> The fermionic or bosonic kernel.

### Returns
* G -> Reconstructed correlator.

See also: [`calc_goodness`](@ref).
"""
function calc_correlator(SE::StochSKElement{I,T}, kernel::Array{T,2}) where {I<:Int,T<:Real}
    pn = length(SE.P)
    𝐴 = fill(SE.A, pn)
    𝐾 = kernel[:, SE.P]
    return 𝐾 * 𝐴
end

"""
	calc_goodness(Gₙ::Vector{F64}, Gᵥ::Vector{F64})

Try to calculate the goodness-of-fit function (i.e, χ²), which measures
the distance between input and regenerated correlators.

### Arguments
* Gₙ -> Reconstructed correlators.
* Gᵥ -> Input (original) correlators.

### Returns
* χ² -> Goodness-of-fit function.

See also: [`calc_correlator`](@ref).
"""
function calc_goodness(Gₙ::Vector{T}, Gᵥ::Vector{T}) where {T<:Real}
    ΔG = Gₙ - Gᵥ
    return dot(ΔG, ΔG)
end

"""
	calc_theta(len::Int, SC::StochSKContext{I,T}, alg::SSK) where {I<:Int,T<:Real}

Try to locate the optimal Θ and χ². This function implements the `chi2min`
and `chi2kink` algorithms. Note that the `chi2min` algorithm is preferred.

### Arguments
* len -> Length of vector Θ.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.
### Returns
* c -> Selected index for optimal Θ.
"""
function calc_theta(len::Int, SC::StochSKContext{I,T}, alg::SSK) where {I<:Int,T<:Real}
    function fitfun(x, p)
        return @. p[1] + p[2] / (1 + exp(-p[4] * (x - p[3])))
    end

    # Which algorithm is preferred ?
    method = alg.method

    # Get length of Θ and χ² vectors
    c = len

    # `chi2min` algorithm, proposed by Shao and Sandvik
    if method == "chi2min"
        while c ≥ 1
            if SC.χ²vec[c] > SC.χ²min + 2 * sqrt(SC.χ²min)
                break
            end
            c = c - 1
        end
    end

    # `chi2kink` algorithm, inspired by the `chi2kink` algorithm
    # used in MaxEnt solver
    if method == "chi2kink"
        guess = [T(0), T(5), T(2), T(0)]
        fit = curve_fit(fitfun, log10.(SC.Θvec[1:c]), log10.(SC.χ²vec[1:c]), guess)
        _, _, a, b = fit.param
        #
        fit_pos = T(5 // 2)
        Θ_opt = a - fit_pos / b
        c = argmin(abs.(log10.(SC.Θvec[1:c]) .- Θ_opt))
    end

    return c
end

#=
"""
	constraints(S::StochSKSolver, fmesh::AbstractMesh)

Try to implement the constrained stochastic analytic continuation
method. This function will return a collection. It contains all the
allowable indices. Be careful, `fmesh` should be a fine linear mesh.

### Arguments
* S     -> A StochSKSolver struct.
* fmesh -> Very dense mesh for the δ peaks.

### Returns
* allow -> Allowable indices.

See also: [`StochSKSolver`](@ref).
"""
function constraints(S::StochSKSolver, fmesh::AbstractMesh)
	exclude = get_b("exclude")
	nfine = get_k("nfine")
	@assert nfine == length(fmesh)

	allow = Int[]

	# Go through the fine mesh and check every mesh point.
	# Is is excluded?
	for i in eachindex(fmesh)
		is_excluded = false
		#
		if !isa(exclude, Missing)
			for j in eachindex(exclude)
				if exclude[j][1] ≤ fmesh[i] ≤ exclude[j][2]
					is_excluded = true
					break
				end
			end
		end
		#
		if !is_excluded
			push!(allow, i)
		end
	end

	return allow
end
=#

"""
	try_move_s!(MC::StochSKMC, SE::StochSKElement, SC::StochSKContext, alg::SSK)

Try to update the Monte Carlo field configurations via the Metropolis
algorithm. In each update, only single δ function is shifted.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
N/A

See also: [`try_move_p!`](@ref).
"""
function try_move_s!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                     alg::SSK) where {I<:Int,T<:Real}
    # Get parameters
    nfine = alg.nfine
    pn = alg.npole

    # Reset counters
    MC.Sacc = 0
    MC.Stry = pn
    @assert 1 < SE.W ≤ nfine

    # Allocate memory for new correlator
    Gₙ = zeros(T, size(SC.Gᵧ))
    ΔG = zeros(T, size(SC.Gᵧ))

    for _ in 1:pn
        # Choose single δ function
        s = rand(MC.rng, 1:pn)

        # Evaluate new position for the δ function
        pcurr = SE.P[s]
        #
        if 1 < SE.W < nfine
            δW = rand(MC.rng, 1:(SE.W))
            #
            if rand(MC.rng) > 0.5
                pnext = pcurr + δW
            else
                pnext = pcurr - δW
            end
            #
            pnext < 1 && (pnext = pnext + nfine)
            pnext > nfine && (pnext = pnext - nfine)
        else
            pnext = rand(MC.rng, 1:nfine)
        end

        # Apply the constraints
        !(pnext in SC.allow) && continue

        # Calculate the transition probability
        Knext = view(SC.kernel, :, pnext)
        Kcurr = view(SC.kernel, :, pcurr)
        #
        @. Gₙ = SC.Gᵧ + SE.A * (Knext - Kcurr)
        @. ΔG = Gₙ - SC.Gᵥ
        χ²new = dot(ΔG, ΔG)
        #
        prob = exp(1 // 2 * (SC.χ² - χ²new) / SC.Θ)

        # Important sampling, if true, the δ function is shifted and the
        # corresponding objects are updated.
        if rand(MC.rng) < min(prob, 1)
            SE.P[s] = pnext
            @. SC.Gᵧ = Gₙ
            #
            SC.χ² = χ²new
            if χ²new < SC.χ²min
                SC.χ²min = χ²new
            end
            #
            MC.Sacc = MC.Sacc + 1
        end
    end
end

"""
	try_move_p!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
			   alg::SSK) where {I<:Int,T<:Real}

Try to update the Monte Carlo field configurations via the Metropolis
algorithm. In each update, only a pair of δ functions are shifted.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
N/A

See also: [`try_move_s!`](@ref).
"""
function try_move_p!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                     alg::SSK) where {I<:Int,T<:Real}
    # Get parameters
    nfine = alg.nfine
    pn = alg.npole

    # We have to make sure that there are at least two δ functions here.
    pn < 2 && return

    # Reset counters
    MC.Pacc = 0
    MC.Ptry = pn
    @assert 1 < SE.W ≤ nfine

    # Allocate memory for new correlator
    Gₙ = zeros(T, size(SC.Gᵧ))
    ΔG = zeros(T, size(SC.Gᵧ))

    for _ in 1:pn
        # Choose a pair of δ functions
        s₁ = rand(MC.rng, 1:pn)
        s₂ = s₁
        while s₁ == s₂
            s₂ = rand(MC.rng, 1:pn)
        end

        # Evaluate new positions for the two δ functions
        pcurr₁ = SE.P[s₁]
        pcurr₂ = SE.P[s₂]
        #
        if 1 < SE.W < nfine
            δW₁ = rand(MC.rng, 1:(SE.W))
            δW₂ = rand(MC.rng, 1:(SE.W))
            #
            if rand(MC.rng) > 0.5
                pnext₁ = pcurr₁ + δW₁
                pnext₂ = pcurr₂ - δW₂
            else
                pnext₁ = pcurr₁ - δW₁
                pnext₂ = pcurr₂ + δW₂
            end
            #
            pnext₁ < 1 && (pnext₁ = pnext₁ + nfine)
            pnext₁ > nfine && (pnext₁ = pnext₁ - nfine)
            pnext₂ < 1 && (pnext₂ = pnext₂ + nfine)
            pnext₂ > nfine && (pnext₂ = pnext₂ - nfine)
        else
            pnext₁ = rand(MC.rng, 1:nfine)
            pnext₂ = rand(MC.rng, 1:nfine)
        end

        # Apply the constraints
        !(pnext₁ in SC.allow) && continue
        !(pnext₂ in SC.allow) && continue

        # Calculate the transition probability
        Knext₁ = view(SC.kernel, :, pnext₁)
        Kcurr₁ = view(SC.kernel, :, pcurr₁)
        Knext₂ = view(SC.kernel, :, pnext₂)
        Kcurr₂ = view(SC.kernel, :, pcurr₂)
        #
        @. Gₙ = SC.Gᵧ + SE.A * (Knext₁ - Kcurr₁ + Knext₂ - Kcurr₂)
        @. ΔG = Gₙ - SC.Gᵥ
        χ²new = dot(ΔG, ΔG)
        #
        prob = exp(1 // 2 * (SC.χ² - χ²new) / SC.Θ)

        # Important sampling, if true, the δ functions are shifted and the
        # corresponding objects are updated.
        if rand(MC.rng) < min(prob, 1)
            SE.P[s₁] = pnext₁
            SE.P[s₂] = pnext₂
            @. SC.Gᵧ = Gₙ
            #
            SC.χ² = χ²new
            if χ²new < SC.χ²min
                SC.χ²min = χ²new
            end
            #
            MC.Pacc = MC.Pacc + 1
        end
    end
end

"""
	try_move_q!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
			   alg::SSK) where {I<:Int,T<:Real}

Try to update the Monte Carlo field configurations via the Metropolis
algorithm. In each update, four different δ functions are shifted.

### Arguments
* MC -> A StochSKMC struct.
* SE -> A StochSKElement struct.
* SC -> A StochSKContext struct.
* alg -> A SSK struct.

### Returns
N/A

See also: [`try_move_s!`](@ref).
"""
function try_move_q!(MC::StochSKMC{I}, SE::StochSKElement{I,T}, SC::StochSKContext{I,T},
                     alg::SSK) where {I<:Int,T<:Real}
    # Get parameters
    nfine = alg.nfine
    pn = alg.npole

    # We have to make sure that there are at least four δ functions here.
    pn < 4 && return

    # Reset counters
    MC.Qacc = 0
    MC.Qtry = pn
    @assert 1 < SE.W ≤ nfine

    # Allocate memory for new correlator
    Gₙ = zeros(T, size(SC.Gᵧ))
    ΔG = zeros(T, size(SC.Gᵧ))

    for _ in 1:pn
        # Choose four different δ functions
        𝑆 = nothing
        while true
            𝑆 = rand(MC.rng, 1:pn, 4)
            𝒮 = unique(𝑆)
            if length(𝑆) == length(𝒮)
                break
            end
        end
        s₁, s₂, s₃, s₄ = 𝑆

        # Evaluate new positions for the four δ functions
        pcurr₁ = SE.P[s₁]
        pcurr₂ = SE.P[s₂]
        pcurr₃ = SE.P[s₃]
        pcurr₄ = SE.P[s₄]
        #
        if 1 < SE.W < nfine
            δW₁ = rand(MC.rng, 1:(SE.W))
            δW₂ = rand(MC.rng, 1:(SE.W))
            δW₃ = rand(MC.rng, 1:(SE.W))
            δW₄ = rand(MC.rng, 1:(SE.W))
            #
            if rand(MC.rng) > 0.5
                pnext₁ = pcurr₁ + δW₁
                pnext₂ = pcurr₂ - δW₂
                pnext₃ = pcurr₃ + δW₃
                pnext₄ = pcurr₄ - δW₄
            else
                pnext₁ = pcurr₁ - δW₁
                pnext₂ = pcurr₂ + δW₂
                pnext₃ = pcurr₃ - δW₃
                pnext₄ = pcurr₄ + δW₄
            end
            #
            pnext₁ < 1 && (pnext₁ = pnext₁ + nfine)
            pnext₁ > nfine && (pnext₁ = pnext₁ - nfine)
            pnext₂ < 1 && (pnext₂ = pnext₂ + nfine)
            pnext₂ > nfine && (pnext₂ = pnext₂ - nfine)
            pnext₃ < 1 && (pnext₃ = pnext₃ + nfine)
            pnext₃ > nfine && (pnext₃ = pnext₃ - nfine)
            pnext₄ < 1 && (pnext₄ = pnext₄ + nfine)
            pnext₄ > nfine && (pnext₄ = pnext₄ - nfine)
        else
            pnext₁ = rand(MC.rng, 1:nfine)
            pnext₂ = rand(MC.rng, 1:nfine)
            pnext₃ = rand(MC.rng, 1:nfine)
            pnext₄ = rand(MC.rng, 1:nfine)
        end

        # Apply the constraints
        !(pnext₁ in SC.allow) && continue
        !(pnext₂ in SC.allow) && continue
        !(pnext₃ in SC.allow) && continue
        !(pnext₄ in SC.allow) && continue

        # Calculate the transition probability
        Knext₁ = view(SC.kernel, :, pnext₁)
        Kcurr₁ = view(SC.kernel, :, pcurr₁)
        Knext₂ = view(SC.kernel, :, pnext₂)
        Kcurr₂ = view(SC.kernel, :, pcurr₂)
        Knext₃ = view(SC.kernel, :, pnext₃)
        Kcurr₃ = view(SC.kernel, :, pcurr₃)
        Knext₄ = view(SC.kernel, :, pnext₄)
        Kcurr₄ = view(SC.kernel, :, pcurr₄)
        #
        @. Gₙ = SC.Gᵧ +
                SE.A * (Knext₁ - Kcurr₁ +
                        Knext₂ - Kcurr₂ +
                        Knext₃ - Kcurr₃ +
                        Knext₄ - Kcurr₄)
        @. ΔG = Gₙ - SC.Gᵥ
        χ²new = dot(ΔG, ΔG)
        #
        prob = exp(1 // 2 * (SC.χ² - χ²new) / SC.Θ)

        # Important sampling, if true, the δ functions are shifted and the
        # corresponding objects are updated.
        if rand(MC.rng) < min(prob, 1)
            SE.P[s₁] = pnext₁
            SE.P[s₂] = pnext₂
            SE.P[s₃] = pnext₃
            SE.P[s₄] = pnext₄
            @. SC.Gᵧ = Gₙ
            #
            SC.χ² = χ²new
            if χ²new < SC.χ²min
                SC.χ²min = χ²new
            end
            #
            MC.Qacc = MC.Qacc + 1
        end
    end
end
