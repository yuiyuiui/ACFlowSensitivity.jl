
"""
    last(brc::BarRatContext)

It will process and write the calculated results by the BarRat solver,
including correlator at real axis, final spectral function, reproduced
correlator. The information about Prony approximation and barycentric
rational function approximation will be written as well.

### Arguments
* brc -> A BarRatContext struct.

### Returns
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function last(brc::BarRatContext)
    # Reconstruct retarded Green's function using pole representation
    function pole_green!(_G::Vector{C64})
        η = get_r("eta")
        if η < 1.0
            # Here we should make sure that the imaginary parts of brc.ℬA
            # and brc.ℬP are quite small. Such that we can ignore them.
            rA = real(brc.ℬA)
            rP = real(brc.ℬP)
            for i in eachindex(_G)
                _G[i] = sum(@. rA / (brc.mesh.mesh[i] - rP + η * im))
            end
        else
            iA = brc.ℬA
            rP = real(brc.ℬP)
            for i in eachindex(_G)
                _G[i] = sum(@. iA / (brc.mesh.mesh[i] - rP + (η - 1) * im))
            end
        end
    end

    # Calculate full response function on real axis and write them
    _G = brc.ℬ.(brc.mesh.mesh)
    get_r("atype") == "delta" && pole_green!(_G)

    # Calculate and write the spectral function
    Aout = -imag.(_G) ./ π

    # Regenerate the input data and write them
    #
    # Be careful, BarRat will always give A(ω), instead of A(ω)/ω. This
    # will lead to problem when we try to reproduce the input data, becase
    # A(ω) is not compatible with the predefined kernel. So we need to
    # convert A(ω) to A(ω)/ω when the system is bosonic.
    kernel = make_kernel(brc.mesh, brc.grid)
    if get_b("ktype") == "fermi"
        G = reprod(brc.mesh, kernel, Aout)
    else
        Aeff = Aout ./ brc.mesh.mesh
        #
        # When ω = 0.0, A(ω)/ω will produce Inf / NaN. We need to avoid this.
        @assert count(z -> isinf(z) || isnan(z), Aeff) == 1
        ind = findfirst(z -> isinf(z) || isnan(z), Aeff)
        #
        if ind == 1
            Aeff[ind] = 2.0 * Aeff[ind+1] - Aeff[ind+2]
        elseif ind == length(Aeff)
            Aeff[ind] = 2.0 * Aeff[ind-1] - Aeff[ind-2]
        else
            Aeff[ind] = (Aeff[ind-1] + Aeff[ind+1]) / 2.0
        end
        #
        G = reprod(brc.mesh, kernel, Aeff)
    end

    return Aout, _G
end

"""
    poles!(brc::BarRatContext)

Convert the barycentric rational function approximation to the classic
pole representation. Note that this feature is only suitable for the
`atype` = "delta" case. In such case, the barycenteric algorithm can find
the accurate positions for the poles via the `bc_poles()` function. But
it seems that the weights for these poles are wrong. In this function, we
just use the BFGS method to solve this optimization problem to get the
correct weights for the poles. And then the positions and weights of these
poles will be stored in `brc`, a BarRatContext struct.

### Arguments
* brc -> A BarRatContext struct.

### Returns
N/A
"""
function poles!(brc::BarRatContext)
    function 𝑓(x::Vector{C64})
        Gₙ = zeros(C64, length(brc.Gᵥ))
        iωₙ = brc.grid.ω * im
        #
        for i in eachindex(x)
            @. Gₙ = Gₙ + x[i] / (iωₙ - brc.ℬP[i])
        end
        #
        return sum(abs.(Gₙ - brc.Gᵥ))
    end

    function 𝐽!(J::Vector{C64}, x::Vector{C64})
        # The Zygote.gradient() fails here.
        return J .= gradient_via_fd(𝑓, x)
    end

    # Get positions of the poles
    𝑃 = bc_poles(brc.ℬ)
    #
    # Print their positions
    println("Raw poles:")
    for i in eachindex(𝑃)
        z = 𝑃[i]
        @printf("P %4i -> %16.12f + %16.12f im \n", i, real(z), imag(z))
    end
    #
    # Filter unphysical poles
    filter!(z -> abs(imag(z)) < get_r("pcut"), 𝑃)
    if length(𝑃) == 0
        error("The number of poles is zero. You should increase pcut")
    end
    #
    # Print their positions again
    println("New poles:")
    for i in eachindex(𝑃)
        z = 𝑃[i]
        @printf("P %4i -> %16.12f + %16.12f im \n", i, real(z), imag(z))
    end
    #
    # Update BarRatContext
    brc.ℬP = 𝑃

    # Now we know positions of these poles, and we need to figure out
    # their amplitudes. This is a typical optimization problem. We just
    # employ the BFGS algorithm to do this job.
    𝐴 = zeros(C64, length(𝑃))
    res = optimize(𝑓, 𝐽!, 𝐴; max_iter=500)
    brc.ℬA = res.minimizer
    #
    # Print their weights / amplitudes.
    println("New poles:")
    for i in eachindex(𝐴)
        z = brc.ℬA[i]
        @printf("A %4i -> %16.12f + %16.12f im \n", i, real(z), imag(z))
    end
    #
    # Well, we should check whether these amplitudes are reasonable.
    #@assert all(z -> abs(imag(z)) < get_r("pcut"), brc.ℬA)
end

"""
    bc_poles(r::BarycentricFunction)

Return the poles of the rational function `r`.

### Arguments
* r -> A BarycentricFunction struct.

### Returns
* pole -> List of poles.
"""
function bc_poles(r::BarycentricFunction)
    w = bc_weights(r)
    z = bc_nodes(r)
    nonzero = @. !iszero(w)
    z, w = z[nonzero], w[nonzero]
    #
    m = length(w)
    B = diagm([zero(F64); ones(F64, m)])
    E = [zero(F64) transpose(w); ones(F64, m) diagm(z)];
    #
    pole = [] # Put it into scope
    try
        pole = filter(isfinite, eigvals(E, B))
    catch
        # Generalized eigen not available in extended precision, so:
        λ = filter(z->abs(z)>1e-13, eigvals(E\B))
        pole = 1 ./ λ
    end

    return pole
end

"""
    BarRatContext

Mutable struct. It is used within the BarRat solver only.

### Members
* Gᵥ   -> Input data for correlator.
* grid -> Grid for input data.
* mesh -> Mesh for output spectrum.
* 𝒫    -> Prony approximation for the input data.
* ℬ    -> Barycentric rational function approximation for the input data.
* ℬP   -> It means the positions of the poles.
* ℬA   -> It means the weights / amplitudes of the poles.
"""
mutable struct BarRatContext
    Gᵥ::Vector{C64}
    grid::AbstractGrid
    mesh::AbstractMesh
    𝒫::Union{Missing,PronyApproximation}
    ℬ::Union{Missing,BarycentricFunction}
    ℬP::Vector{C64}
    ℬA::Vector{C64}
end
