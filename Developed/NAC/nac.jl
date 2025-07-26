mutable struct NevanACContext{T<:Real,I<:Int}
    Gᵥ::Vector{APC}
    grid::AbstractGrid
    mesh::AbstractMesh
    Φ::Vector{APC}
    𝒜::Array{APC,3}
    ℋ::Array{APC,2}
    𝑎𝑏::Vector{Complex{T}}
    hmin::I
    hopt::I
end

function solve(GFV::Vector{Complex{T}}, ctx::ContextData{T}, alg::NevanAC) where {T<:Real}
    println("[ NevanAC ]")
    nac = init(S, rd)
    run(nac)
    Aout, _ = last(nac)
    return nac.mesh.mesh, Aout
end

function init(S::NevanACSolver, rd::RawData)
    # Setup numerical precision. Note that the NAC method is extremely
    # sensitive to the float point precision.
    setprecision(128)

    # Convert the input data to APC, i.e., Complex{BigFloat}.
    ωₙ = APC.(rd._grid * im)
    Gₙ = APC.(rd.value)

    # Evaluate the optimal value for the size of input data.
    # Here we just apply the Pick criterion.
    ngrid = calc_noptim(ωₙ, Gₙ)

    # Prepera input data
    Gᵥ = calc_mobius(-Gₙ[1:ngrid])
    reverse!(Gᵥ)
    println("Postprocess input data: ", length(Gᵥ), " points")

    # Prepare grid for input data
    grid = make_grid(rd; T=APF)
    resize!(grid, ngrid)
    reverse!(grid)
    println("Build grid for input data: ", length(grid), " points")

    # Prepare mesh for output spectrum
    mesh = make_mesh(; T=APF)
    println("Build mesh for spectrum: ", length(mesh), " points")

    # Precompute key quantities to accelerate the computation
    Φ, 𝒜, ℋ, 𝑎𝑏 = precompute(grid, mesh, Gᵥ)
    println("Precompute key matrices")

    # Actually, the NevanACContext struct already contains enough
    # information to build the Nevanlinna interpolant and get the
    # spectrum, but Hardy basis optimization is needed to smooth
    # the results further.
    return NevanACContext(Gᵥ, grid, mesh, Φ, 𝒜, ℋ, 𝑎𝑏, 1, 1)
end

function run(nac::NevanACContext)
    hardy = get_n("hardy")
    #
    if hardy
        println("Activate Hardy basis optimization")

        # Determine the minimal Hardy order (`hmin`), update `ℋ` and `𝑎𝑏`.
        calc_hmin!(nac)

        # Determine the optimal Hardy order (`hopt`), update `ℋ` and `𝑎𝑏`.
        calc_hopt!(nac)
    end
end

function last(nac::NevanACContext)
    # By default, we should write the analytic continuation results
    # into the external files.
    _fwrite = get_b("fwrite")
    fwrite = isa(_fwrite, Missing) || _fwrite ? true : false

    # Calculate full response function on real axis and write them
    # Note that _G is actually 𝑁G, so there is a `-` symbol for the
    # return value.
    _G = C64.(calc_green(nac.𝒜, nac.ℋ, nac.𝑎𝑏))
    fwrite && write_complete(nac.mesh, -_G)

    # Calculate and write the spectral function
    Aout = F64.(imag.(_G) ./ π)
    fwrite && write_spectrum(nac.mesh, Aout)

    # Regenerate the input data and write them
    kernel = make_kernel(nac.mesh, nac.grid)
    G = reprod(nac.mesh, kernel, Aout)
    fwrite && write_backward(nac.grid, G)

    return Aout, -_G
end

function precompute(grid::AbstractGrid,
                    mesh::AbstractMesh,
                    Gᵥ::Vector{APC})
    # Evaluate ϕ and `abcd` matrices
    Φ = calc_phis(grid, Gᵥ)
    𝒜 = calc_abcd(grid, mesh, Φ)

    # Allocate memory for evaluating θ
    # The initial Hardy order is just 1.
    ℋ = calc_hmatrix(mesh, 1)
    𝑎𝑏 = zeros(C64, 2)

    return Φ, 𝒜, ℋ, 𝑎𝑏
end

function calc_mobius(z::Vector{APC})
    return @. (z - im) / (z + im)
end

function calc_inv_mobius(z::Vector{APC})
    return @. im * (one(APC) + z) / (one(APC) - z)
end

function calc_pick(k::I64, ℎ::Vector{APC}, λ::Vector{APC})
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

function calc_phis(grid::AbstractGrid, Gᵥ::Vector{APC})
    ngrid = length(grid)

    # Allocate memory
    Φ = zeros(APC, ngrid)
    𝒜 = zeros(APC, 2, 2, ngrid)
    ∏ = zeros(APC, 2, 2)
    𝑔 = grid.ω * im

    # Initialize the `abcd` matrix
    for i in 1:ngrid
        𝒜[:, :, i] .= Matrix{APC}(I, 2, 2)
    end

    # Evaluate Φ using recursive algorithm
    Φ[1] = Gᵥ[1]
    for j in 1:(ngrid - 1)
        for k in (j + 1):ngrid
            ∏[1, 1] = (𝑔[k] - 𝑔[j]) / (𝑔[k] - conj(𝑔[j]))
            ∏[1, 2] = Φ[j]
            ∏[2, 1] = conj(Φ[j]) * ∏[1, 1]
            ∏[2, 2] = one(APC)
            view(𝒜,:,:,k) .= view(𝒜,:,:,k) * ∏
        end
        num = 𝒜[1, 2, j+1] - 𝒜[2, 2, j+1] * Gᵥ[j+1]
        den = 𝒜[2, 1, j+1] * Gᵥ[j+1] - 𝒜[1, 1, j+1]
        Φ[j+1] = num / den
    end

    return Φ
end

function calc_abcd(grid::AbstractGrid, mesh::AbstractMesh, Φ::Vector{APC})
    eta::APF = get_n("eta")

    ngrid = length(grid)
    nmesh = length(mesh)

    𝑔 = grid.ω * im
    𝑚 = mesh.mesh .+ im * eta

    𝒜 = zeros(APC, 2, 2, nmesh)
    ∏ = zeros(APC, 2, 2)

    for i in 1:nmesh
        result = Matrix{APC}(I, 2, 2)
        𝑧 = 𝑚[i]
        for j in 1:ngrid
            ∏[1, 1] = (𝑧 - 𝑔[j]) / (𝑧 - conj(𝑔[j]))
            ∏[1, 2] = Φ[j]
            ∏[2, 1] = conj(Φ[j]) * ∏[1, 1]
            ∏[2, 2] = one(APC)
            result *= ∏
        end

        𝒜[:, :, i] .= result
    end

    return 𝒜
end

function calc_hbasis(z::APC, k::I64)
    w = (z - im) / (z + im)
    return 1.0 / (sqrt(π) * (z + im)) * w^k
end

function calc_hmatrix(mesh::AbstractMesh, H::I64)
    # Build real axis
    eta::APF = get_n("eta")
    𝑚 = mesh.mesh .+ eta * im

    # Allocate memory for the Hardy matrix
    nmesh = length(mesh)
    ℋ = zeros(APC, nmesh, 2*H)

    # Build the Hardy matrix
    for k in 1:H
        ℋ[:, 2*k-1] .= calc_hbasis.(𝑚, k-1)
        ℋ[:, 2*k] .= conj(ℋ[:, 2*k-1])
    end

    return ℋ
end

function calc_theta(𝒜::Array{APC,3}, ℋ::Array{APC,2}, 𝑎𝑏::Vector{C64})
    # Well, we should calculate θₘ₊₁ at first.
    θₘ₊₁ = ℋ * 𝑎𝑏

    # Then we evaluate θ according Eq. (7)
    num = 𝒜[1, 1, :] .* θₘ₊₁ .+ 𝒜[1, 2, :]
    den = 𝒜[2, 1, :] .* θₘ₊₁ .+ 𝒜[2, 2, :]
    θ = num ./ den

    return θ
end

function calc_green(𝒜::Array{APC,3}, ℋ::Array{APC,2}, 𝑎𝑏::Vector{C64})
    θ = calc_theta(𝒜, ℋ, 𝑎𝑏)
    gout = calc_inv_mobius(θ)

    return gout
end

function calc_noptim(ωₙ::Vector{APC}, Gₙ::Vector{APC})
    # Get size of input data
    ngrid = length(ωₙ)

    # Check whether the Pick criterion is applied
    pick = get_n("pick")
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

function calc_hmin!(nac::NevanACContext)
    hmax = get_n("hmax")

    h = 1
    while h ≤ hmax
        println("H (Order of Hardy basis) -> $h")

        # Prepare initial ℋ and 𝑎𝑏
        ℋ = calc_hmatrix(nac.mesh, h)
        𝑎𝑏 = zeros(C64, 2*h)

        # Hardy basis optimization
        causality, optim = hardy_optimize!(nac, ℋ, 𝑎𝑏, h)

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

function calc_hopt!(nac::NevanACContext)
    hmax = get_n("hmax")

    for h in (nac.hmin + 1):hmax
        println("H (Order of Hardy basis) -> $h")

        # Prepare initial ℋ and 𝑎𝑏
        ℋ = calc_hmatrix(nac.mesh, h)
        𝑎𝑏 = copy(nac.𝑎𝑏)
        push!(𝑎𝑏, zero(C64))
        push!(𝑎𝑏, zero(C64))
        @assert size(ℋ)[2] == length(𝑎𝑏)

        # Hardy basis optimization
        causality, optim = hardy_optimize!(nac, ℋ, 𝑎𝑏, h)

        # Check whether the causality is preserved and the
        # optimization is successful.
        if !(causality && optim)
            break
        end
    end
end

function hardy_optimize!(nac::NevanACContext,
                         ℋ::Array{APC,2},
                         𝑎𝑏::Vector{C64},
                         H::I64)
    # Function call to the smooth norm.
    function 𝑓(x::Vector{C64})
        return smooth_norm(nac, ℋ, x)
    end

    # Function call to the gradient of the smooth norm.
    #
    # Here we adopt the Zygote package, which implements an automatic
    # differentiation algorithm, to evaluate the gradient of the smooth
    # norm. Of course, we can turn to the finite difference algorithm,
    # which is less efficient.
    function 𝐽!(J::Vector{C64}, x::Vector{C64})
        #J .= Zygote.gradient(𝑓, x)[1]

        # Finite difference algorithm
        return J .= gradient_via_fd(𝑓, x)
    end

    # Perform numerical optimization by the BFGS algorithm.
    # If it is failed, please turn to the Optim.jl package.
    # A simplified version is implemented in math.jl.
    res = optimize(𝑓, 𝐽!, 𝑎𝑏; max_iter=500)

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

function smooth_norm(nac::NevanACContext, ℋ::Array{APC,2}, 𝑎𝑏::Vector{C64})
    # Get regulation parameter
    α = get_n("alpha")

    # Generate output spectrum
    _G = calc_green(nac.𝒜, ℋ, 𝑎𝑏)
    A = F64.(imag.(_G) ./ π)

    # Normalization term
    𝑓₁ = trapz(nac.mesh, A)

    # Smoothness term
    sd = second_derivative(nac.mesh.mesh, A)
    x_sd = nac.mesh.mesh[2:(end - 1)]
    𝑓₂ = trapz(x_sd, abs.(sd) .^ 2)

    # Assemble the final smooth norm
    𝐹 = abs(1.0 - 𝑓₁)^2 + α * 𝑓₂

    return F64(𝐹)
end

function check_pick(wn::Vector{APC}, gw::Vector{APC}, Nopt::I64)
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

function check_causality(ℋ::Array{APC,2}, 𝑎𝑏::Vector{C64})
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
