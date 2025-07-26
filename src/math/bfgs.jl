
"""
    BFGSDifferentiable

Mutable struct. It is used for objectives and solvers where the gradient
is available/exists.

### Members
* ℱ! -> Objective. It is actually a function call and return objective.
* 𝒟! -> It is a function call as well and returns derivative of objective.
* 𝐹  -> Cache for ℱ! output.
* 𝐷  -> Cache for 𝒟! output.
"""
mutable struct BFGSDifferentiable
    ℱ!::Any
    𝒟!::Any
    𝐹::Any
    𝐷::Any
end

"""
    BFGSDifferentiable(f, df, x::AbstractArray)

Constructor for BFGSDifferentiable struct. `f` is the function, `df` is
the derivative of objective, `x` is the initial guess.
"""
function BFGSDifferentiable(f, df, x::AbstractArray)
    𝐹 = real(zero(eltype(x)))
    T = promote_type(eltype(x), eltype(𝐹))
    𝐷 = fill!(T.(x), T(NaN))
    return BFGSDifferentiable(f, df, copy(𝐹), copy(𝐷))
end

"""
    value(obj::BFGSDifferentiable)

Return `obj.𝐹`. `obj` will not be affected.
"""
value(obj::BFGSDifferentiable) = obj.𝐹

"""
    bfgsgrad(obj::BFGSDifferentiable)

Return `obj.𝐷`. `obj` will not be affected.
"""
bfgsgrad(obj::BFGSDifferentiable) = obj.𝐷

"""
    value_gradient!(obj::BFGSDifferentiable, x)

Evaluate objective and derivative at `x`. `obj.𝐹` and `obj.𝐷` should be
updated. Note that here `obj.𝒟!` is actually `nac.jl/smooth_norm()`.
"""
function value_gradient!(obj::BFGSDifferentiable, x)
    # Note that gradient(obj), i.e obj.𝐷, should be updated in obj.𝒟!().
    obj.𝒟!(bfgsgrad(obj), x)
    return obj.𝐹 = obj.ℱ!(x)
end

"""
    BFGSState

Mutable struct. It is used to trace the history of states visited.

### Members
* x     -> Current position.
* ls    -> Current search direction.
* δx    -> Changes in position.
* δg    -> Changes in gradient.
* xₚ    -> Previous position.
* gₚ    -> Previous gradient.
* fₚ    -> Previous f (f in xₚ).
* H⁻¹   -> Current inverse Hessian matrix.
* alpha -> A internal parameter to control the BFGS algorithm.
"""
mutable struct BFGSState{Tx,Tm,T,G}
    x::Tx
    ls::Tx
    δx::Tx
    δg::Tx
    xₚ::Tx
    gₚ::G
    fₚ::T
    H⁻¹::Tm
    alpha::T
end

"""
    BFGSOptimizationResults

It is used to save the optimization results of the BFGS algorithm.

### Members
* x₀         -> Initial guess for the solution.
* minimizer  -> Final results for the solution.
* minimum    -> Objective at the final solution.
* iterations -> Number of iterations.
* δx         -> Absolute change in x.
* Δx         -> Relative change in x.
* δf         -> Absolute change in f.
* Δf         -> Relative change in f.
* resid      -> Maximum gradient of f at the final solution.
* gconv      -> If the convergence criterion is satisfied
"""
mutable struct BFGSOptimizationResults{Tx,Tc,Tf}
    x₀::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    δx::Tc
    Δx::Tc
    δf::Tc
    Δf::Tc
    resid::Tc
    gconv::Bool
end

"""
    maxdiff(x::AbstractArray, y::AbstractArray)

Return the maximum difference between two arrays: `x` and `y`. Note that
the sizes of `x` and `y` should be the same.
"""
function maxdiff(x::AbstractArray, y::AbstractArray)
    return mapreduce((a, b) -> abs(a - b), max, x, y)
end

"""
    eval_δf(d::BFGSDifferentiable, s::BFGSState)

Evaluate the absolute changes in f.
"""
eval_δf(d::BFGSDifferentiable, s::BFGSState) = abs(value(d) - s.fₚ)

"""
    eval_Δf(d::BFGSDifferentiable, s::BFGSState)

Evaluate the relative changes in f.
"""
eval_Δf(d::BFGSDifferentiable, s::BFGSState) = eval_δf(d, s) / abs(value(d))

"""
    eval_δx(s::BFGSState)

Evaluate the absolute changes in x.
"""
eval_δx(s::BFGSState) = maxdiff(s.x, s.xₚ)

"""
    eval_Δx(s::BFGSState)

Evaluate the relative changes in x.
"""
eval_Δx(s::BFGSState) = eval_δx(s) / maximum(abs, s.x)

"""
    eval_resid(d::BFGSDifferentiable)

Evaluate residual (maximum gradient of f at the current position).
"""
eval_resid(d::BFGSDifferentiable) = maximum(abs, bfgsgrad(d))

"""
    bfgs(f, g, x₀::AbstractArray; max_iter::Int = 1000)

Return the argmin over x of `f(x)` using the BFGS algorithm. Here, `f`
is the function call, and `g` will return the gradient of `f`, `x₀` is
an initial guess for the solution.
"""
function bfgs(f, g, x₀::AbstractArray; max_iter::Int=1000)
    # Initialize time stamp
    t₀ = time()

    # Create BFGSDifferentiable
    d = BFGSDifferentiable(f, g, x₀)

    # Create BFGSState
    s = init_state(d, x₀)

    # Prepare counter
    iteration = 0

    # Print trace for optimization progress
    println("Tracing BFGS Optimization")
    trace(d, iteration, time() - t₀)

    # Setup convergence flag
    gconv = !isfinite(value(d)) || any(!isfinite, bfgsgrad(d))
    while !gconv && iteration < max_iter
        iteration += 1

        # Update line search direction
        ls_success = !update_state!(d, s)
        if !ls_success
            break
        end

        # Update the value of f and its gradient
        update_g!(d, s)

        # Update the Hessian matrix
        update_h!(d, s)

        # Print trace for optimization progress
        trace(d, iteration, time() - t₀)

        # Check the gradient
        if !all(isfinite, bfgsgrad(d))
            @warn "Terminated early due to NaN in gradient."
            break
        end

        # Check whether convergence criterion is satisfied
        gconv = (eval_resid(d) ≤ 1e-8)
        @show gconv
    end

    # Return BFGSOptimizationResults
    return BFGSOptimizationResults(x₀, s.x, value(d), iteration,
                                   eval_δx(s), eval_Δx(s),
                                   eval_δf(d, s), eval_Δf(d, s),
                                   eval_resid(d),
                                   gconv)
end

"""
    init_state(d::BFGSDifferentiable, x₀::AbstractArray)

Create a BFGSState object. Note that `d` should be updated in this
function (`d.𝐹` and `d.𝐷`). `x₀` is an initial guess for the solution.

See also: [`BFGSDifferentiable`](@ref), [`BFGSState`](@ref).
"""
function init_state(d::BFGSDifferentiable, x₀::AbstractArray)
    # Update `d.𝐹` and `d.𝐷` using x₀.
    value_gradient!(d, x₀)

    # Prepare inverse Hessian matrix
    T = eltype(x₀)
    x_ = reshape(x₀, :)
    H⁻¹ = x_ .* x_' .* false
    idxs = diagind(H⁻¹)
    scale = T(1)
    @. @view(H⁻¹[idxs]) = scale * true

    # Return BFGSState
    return BFGSState(x₀, similar(x₀), similar(x₀), similar(x₀), copy(x₀),
                     copy(bfgsgrad(d)), real(T)(NaN),
                     H⁻¹,
                     real(one(T)))
end

"""
    update_state!(d::BFGSDifferentiable, s::BFGSState)

Evaluate line search direction and change of x. New position and old
gradient are saved in `s`.

See also: [`BFGSDifferentiable`](@ref), [`BFGSState`](@ref).
"""
function update_state!(d::BFGSDifferentiable, s::BFGSState)
    T = eltype(s.ls)

    # Set the search direction
    #
    # Note that search direction is the negative gradient divided by
    # the approximate Hessian matrix.
    mul!(vec(s.ls), s.H⁻¹, vec(bfgsgrad(d)))
    rmul!(s.ls, T(-1))

    # Maintain a record of the previous gradient
    copyto!(s.gₚ, bfgsgrad(d))

    # Determine the distance of movement along the search line
    lssuccess = linesearch!(d, s)

    # Update current position
    s.δx .= s.alpha .* s.ls
    s.x .= s.x .+ s.δx

    # Break on linesearch error
    return lssuccess == false
end

"""
    update_g!(d::BFGSDifferentiable, s::BFGSState)

Update the function value and gradient (`d.𝐹` and `d.𝐷` are changed).

See also: [`BFGSDifferentiable`](@ref), [`BFGSState`](@ref).
"""
update_g!(d::BFGSDifferentiable, s::BFGSState) = value_gradient!(d, s.x)

"""
    update_h!(d::BFGSDifferentiable, s::BFGSState)

Try to evaluate the new Hessian matrix.

See also: [`BFGSDifferentiable`](@ref), [`BFGSState`](@ref).
"""
function update_h!(d::BFGSDifferentiable, s::BFGSState)
    n = length(s.x)
    su = similar(s.x)

    # Measure the change in the gradient
    s.δg .= bfgsgrad(d) .- s.gₚ

    # Update the inverse Hessian approximation by using the
    # famous Sherman-Morrison equation
    dx_dg = real(dot(s.δx, s.δg))
    if dx_dg > 0
        mul!(vec(su), s.H⁻¹, vec(s.δg))

        c1 = (dx_dg + real(dot(s.δg, su))) / (dx_dg' * dx_dg)
        c2 = 1 / dx_dg

        # H⁻¹ = H⁻¹ + c1 * (s * s') - c2 * (u * s' + s * u')
        if s.H⁻¹ isa Array
            H⁻¹ = s.H⁻¹
            dx = s.δx
            u = su
            @inbounds for j in 1:n
                c1dxj = c1 * dx[j]'
                c2dxj = c2 * dx[j]'
                c2uj = c2 * u[j]'
                for i in 1:n
                    H⁻¹[i, j] = muladd(dx[i], c1dxj,
                                       muladd(-u[i], c2dxj,
                                              muladd(c2uj, -dx[i], H⁻¹[i, j])))
                end
            end
        else
            mul!(s.H⁻¹, vec(s.δx), vec(s.δx)', +c1, 1)
            mul!(s.H⁻¹, vec(su), vec(s.δx)', -c2, 1)
            mul!(s.H⁻¹, vec(s.δx), vec(su)', -c2, 1)
        end
    end
end

"""
    trace(d::BFGSDifferentiable, iter::I, curr_time::T) where {I,T}

Print some useful information about the optimization procedure.
"""
function trace(d::BFGSDifferentiable, iter::I, curr_time::T) where {I,T}
    gnorm = norm(bfgsgrad(d), Inf)
    #
    println("iter = $iter, 𝑓 = $(value(d)), ||∂𝑓/∂x|| = $gnorm, time = $(curr_time) (s)")
    #
    return flush(stdout)
end

"""
    linesearch!(d::BFGSDifferentiable, s::BFGSState)

Evaluate line search direction. Actually, `s.alpha`, `s.fₚ`, and `s.xₚ`
will be updated in this function.

See also: [`BFGSDifferentiable`](@ref), [`BFGSState`](@ref).
"""
function linesearch!(d::BFGSDifferentiable, s::BFGSState)
    # Calculate search direction dϕ₀
    dϕ₀ = real(dot(bfgsgrad(d), s.ls))

    # Reset the direction if it becomes corrupted
    if dϕ₀ >= zero(dϕ₀)
        dϕ₀ = real(dot(bfgsgrad(d), s.ls))
    end

    # Guess an alpha
    LS(s, 1, false)

    # Store current x and f(x) for next iteration
    ϕ₀ = value(d)
    s.fₚ = ϕ₀
    copyto!(s.xₚ, s.x)

    # Perform line search using the Hager-Zhang algorithm
    try
        s.alpha, _ = LS(d, s.x, s.ls, s.alpha, ϕ₀, dϕ₀)
        return true # lssuccess = true
    catch ex
        # Catch LineSearchException to allow graceful exit
        if isa(ex, LineSearchException)
            s.alpha = ex.alpha
            return false # lssuccess = false
        else
            rethrow(ex)
        end
    end
end

"""
    converged(r::BFGSOptimizationResults)

Check whether the optimization is converged.

See also: [`BFGSOptimizationResults`](@ref).
"""
function converged(r::BFGSOptimizationResults)
    conv_flags = r.gconv
    x_isfinite = isfinite(r.δx) || isnan(r.Δx)
    f_isfinite = if r.iterations > 0
        isfinite(r.δf) || isnan(r.Δf)
    else
        true
    end
    g_isfinite = isfinite(r.resid)
    return conv_flags && all((x_isfinite, f_isfinite, g_isfinite))
end

#=
### *Math* : *Line Search*
=#

#
# Conjugate gradient line search implementation from:
#   W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a
#     conjugate gradient method with guaranteed descent. ACM
#     Transactions on Mathematical Software 32: 113–137.
#

#
# The following codes are borrowed from:
#     https://github.com/JuliaNLSolvers/LineSearches.jl
#

"""
    LineSearchException

Mutable struct. It contains information about the error occured in
the line search.

### Members
* message -> Error message.
* alpha   -> A key parameter used to control line search.
"""
mutable struct LineSearchException{T<:Real} <: Exception
    message::AbstractString
    alpha::T
end

"""
    LS(state::BFGSState, alpha::T, scaled::Bool)

Line search: initial and static version.
"""
function LS(state::BFGSState, alpha::T, scaled::Bool) where {T}
    PT = promote_type(T, real(eltype(state.ls)))
    if scaled == true && (ns = real(norm(state.ls))) > convert(PT, 0)
        state.alpha = convert(PT, min(alpha, ns)) / ns
    else
        state.alpha = convert(PT, alpha)
    end
end

"""
    LS(df::BFGSDifferentiable,
        x::Vector{S}, s::Vector{S},
        c::T, phi_0::T, dphi_0::T) where {S, T}

Line search: Hager-Zhang algorithm.
"""
function LS(df::BFGSDifferentiable,
            x::Vector{S}, s::Vector{S},
            c::T, phi_0::T, dphi_0::T) where {S,T}
    delta = T(0.1)
    sigma = T(0.9)
    alphamax = Inf
    rho = T(5.0)
    epsilon = T(1e-6)
    gamma = T(0.66)
    linesearchmax = 50
    psi3 = T(0.1)
    mayterminate = Ref{Bool}(false)

    ϕdϕ = make_ϕdϕ(df, similar(x), x, s)

    zeroT = convert(T, 0)
    #
    if !(isfinite(phi_0) && isfinite(dphi_0))
        throw(LineSearchException("Value and slope at step length = 0 must be finite.",
                                  T(0)))
    end
    #
    if dphi_0 >= eps(T) * abs(phi_0)
        throw(LineSearchException("Search direction is not a direction of descent.", T(0)))
    elseif dphi_0 >= 0
        return zeroT, phi_0
    end

    # Prevent values of x_new = x+αs that are likely to
    # make ϕ(x_new) infinite
    iterfinitemax::Int = ceil(Int, -log2(eps(T)))
    alphas = [zeroT] # for bisection
    values = [phi_0]
    slopes = [dphi_0]

    phi_lim = phi_0 + epsilon * abs(phi_0)
    @assert c >= 0
    c <= eps(T) && return zeroT, phi_0
    @assert isfinite(c) && c <= alphamax
    phi_c, dphi_c = ϕdϕ(c)
    iterfinite = 1
    #
    while !(isfinite(phi_c) && isfinite(dphi_c)) && iterfinite < iterfinitemax
        mayterminate[] = false
        iterfinite += 1
        c *= psi3
        phi_c, dphi_c = ϕdϕ(c)
    end
    #
    if !(isfinite(phi_c) && isfinite(dphi_c))
        @warn("Failed to achieve finite new evaluation point, using alpha=0")
        mayterminate[] = false # reset in case another initial guess is used next
        return zeroT, phi_0
    end
    #
    push!(alphas, c)
    push!(values, phi_c)
    push!(slopes, dphi_c)

    # If c was generated by quadratic interpolation, check whether it
    # satisfies the Wolfe conditions
    if mayterminate[] &&
       satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
        # Reset in case another initial guess is used next
        mayterminate[] = false
        return c, phi_c # phi_c
    end

    # Initial bracketing step (HZ, stages B0-B3)
    isbracketed = false
    ia = 1
    ib = 2
    @assert length(alphas) == 2
    iter = 1
    cold = -one(T)
    while !isbracketed && iter < linesearchmax
        if dphi_c >= zeroT
            # We've reached the upward slope, so we have b; examine
            # previous values to find a
            ib = length(alphas)
            for i in (ib - 1):-1:1
                if values[i] <= phi_lim
                    ia = i
                    break
                end
            end
            isbracketed = true
        elseif values[end] > phi_lim
            # The value is higher, but the slope is downward, so we must
            # have crested over the peak. Use bisection.
            ib = length(alphas)
            ia = 1
            if c ≉ alphas[ib] || slopes[ib] >= zeroT
                error("c = ", c)
            end
            ia, ib = ls_bisect!(ϕdϕ, alphas, values, slopes, ia, ib, phi_lim)
            isbracketed = true
        else
            # We'll still going downhill, expand the interval and try again.
            # Reaching this branch means that dphi_c < 0 and phi_c <= phi_0 + ϵ_k
            # So cold = c has a lower objective than phi_0 up to epsilon.
            # This makes it a viable step to return if bracketing fails.

            # Bracketing can fail if no cold < c <= alphamax can be found
            # with finite phi_c and dphi_c. Going back to the loop with
            # c = cold will only result in infinite cycling. So returning
            # (cold, phi_cold) and exiting the line search is the best move.
            cold = c
            phi_cold = phi_c
            if nextfloat(cold) >= alphamax
                # Reset in case another initial guess is used next
                mayterminate[] = false
                return cold, phi_cold
            end
            c *= rho
            if c > alphamax
                c = alphamax
            end
            phi_c, dphi_c = ϕdϕ(c)
            iterfinite = 1
            while !(isfinite(phi_c) && isfinite(dphi_c)) &&
                      c > nextfloat(cold) && iterfinite < iterfinitemax
                # Shrinks alphamax, assumes that steps >= c can never
                # have finite phi_c and dphi_c.
                alphamax = c
                iterfinite += 1
                c = (cold + c) / 2
                phi_c, dphi_c = ϕdϕ(c)
            end
            if !(isfinite(phi_c) && isfinite(dphi_c))
                return cold, phi_cold
            end
            push!(alphas, c)
            push!(values, phi_c)
            push!(slopes, dphi_c)
        end
        iter += 1
    end

    while iter < linesearchmax
        a = alphas[ia]
        b = alphas[ib]
        @assert b > a
        if b - a <= eps(b)
            # Reset in case another initial guess is used next
            mayterminate[] = false
            return a, values[ia] # lsr.value[ia]
        end
        iswolfe, iA, iB = ls_secant2!(ϕdϕ, alphas, values, slopes, ia, ib, phi_lim, delta,
                                      sigma)
        if iswolfe
            # Reset in case another initial guess is used next
            mayterminate[] = false
            return alphas[iA], values[iA]
        end
        A = alphas[iA]
        B = alphas[iB]
        @assert B > A
        if B - A < gamma * (b - a)
            if nextfloat(values[ia]) >= values[ib] && nextfloat(values[iA]) >= values[iB]
                # It's so flat, secant didn't do anything useful, time to quit
                # Reset in case another initial guess is used next
                mayterminate[] = false
                return A, values[iA]
            end
            ia = iA
            ib = iB
        else
            # Secant is converging too slowly, use bisection
            c = (A + B) / convert(T, 2)

            phi_c, dphi_c = ϕdϕ(c)
            @assert isfinite(phi_c) && isfinite(dphi_c)
            push!(alphas, c)
            push!(values, phi_c)
            push!(slopes, dphi_c)

            ia, ib = ls_update!(ϕdϕ, alphas, values, slopes, iA, iB, length(alphas),
                                phi_lim)
        end
        iter += 1
    end

    throw(LineSearchException("Linesearch failed to converge, reached maximum
                               iterations $(linesearchmax).", alphas[ia]))
end

function make_ϕdϕ(df::BFGSDifferentiable, x_new, x, s)
    function ϕdϕ(α)
        # Move a distance of alpha in the direction of s
        x_new .= x .+ α .* s

        # Evaluate ∇f(x+α*s)
        value_gradient!(df, x_new)

        # Calculate ϕ'(a_i)
        return value(df), real(dot(bfgsgrad(df), s))
    end
    return ϕdϕ
end

# Check Wolfe & approximate Wolfe
function satisfies_wolfe(c::T, phi_c::T, dphi_c::T,
                         phi_0::T, dphi_0::T, phi_lim::T,
                         delta::T, sigma::T) where {T}
    wolfe1 = delta * dphi_0 >= (phi_c - phi_0) / c &&
             dphi_c >= sigma * dphi_0
    wolfe2 = (2 * delta - 1) * dphi_0 >= dphi_c >= sigma * dphi_0 &&
             phi_c <= phi_lim
    return wolfe1 || wolfe2
end

# HZ, stage U3 (with theta=0.5)
function ls_bisect!(ϕdϕ, alphas::Vector{T}, values::Vector{T},
                    slopes::Vector{T}, ia::I, ib::I, phi_lim::T) where {I,T}
    gphi = convert(T, NaN)
    a = alphas[ia]
    b = alphas[ib]

    # Debugging (HZ, conditions shown following U3)
    zeroT = convert(T, 0)
    #
    @assert slopes[ia] < zeroT
    @assert values[ia] <= phi_lim
    @assert slopes[ib] < zeroT
    @assert values[ib] > phi_lim
    @assert b > a
    #
    while b - a > eps(b)
        d = (a + b) / convert(T, 2)
        phi_d, gphi = ϕdϕ(d)
        @assert isfinite(phi_d) && isfinite(gphi)

        push!(alphas, d)
        push!(values, phi_d)
        push!(slopes, gphi)

        id = length(alphas)
        #
        if gphi >= zeroT
            return ia, id # replace b, return
        end
        #
        if phi_d <= phi_lim
            a = d # replace a, but keep bisecting until dphi_b > 0
            ia = id
        else
            b = d
            ib = id
        end
    end

    return ia, ib
end

# HZ, stages S1-S4
function ls_secant(a::T, b::T, dphi_a::T, dphi_b::T) where {T}
    return (a * dphi_b - b * dphi_a) / (dphi_b - dphi_a)
end

function ls_secant2!(ϕdϕ, alphas::Vector{T},
                     values::Vector{T}, slopes::Vector{T},
                     ia::I, ib::I,
                     phi_lim::T, delta::T, sigma::T) where {I,T}
    phi_0 = values[1]
    dphi_0 = slopes[1]
    a = alphas[ia]
    b = alphas[ib]
    dphi_a = slopes[ia]
    dphi_b = slopes[ib]

    zeroT = convert(T, 0)
    #
    if !(dphi_a < zeroT && dphi_b >= zeroT)
        error(string("Search direction is not a direction of descent; ",
                     "this error may indicate that user-provided derivatives are inaccurate. ",
                     "(dphi_a = $(dphi_a); dphi_b = $(dphi_b))"))
    end
    #
    c = ls_secant(a, b, dphi_a, dphi_b)
    @assert isfinite(c)
    #
    phi_c, dphi_c = ϕdϕ(c)
    @assert isfinite(phi_c) && isfinite(dphi_c)

    push!(alphas, c)
    push!(values, phi_c)
    push!(slopes, dphi_c)

    ic = length(alphas)
    if satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
        return true, ic, ic
    end

    iA, iB = ls_update!(ϕdϕ, alphas, values, slopes, ia, ib, ic, phi_lim)
    a = alphas[iA]
    b = alphas[iB]

    if iB == ic
        # We updated b, make sure we also update a
        c = ls_secant(alphas[ib], alphas[iB], slopes[ib], slopes[iB])
    elseif iA == ic
        # We updated a, do it for b too
        c = ls_secant(alphas[ia], alphas[iA], slopes[ia], slopes[iA])
    end
    #
    if (iA == ic || iB == ic) && a <= c <= b
        phi_c, dphi_c = ϕdϕ(c)
        @assert isfinite(phi_c) && isfinite(dphi_c)

        push!(alphas, c)
        push!(values, phi_c)
        push!(slopes, dphi_c)

        ic = length(alphas)
        # Check arguments here
        if satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
            return true, ic, ic
        end
        iA, iB = ls_update!(ϕdϕ, alphas, values, slopes, iA, iB, ic, phi_lim)
    end

    return false, iA, iB
end

# HZ, stages U0-U3
#
# Given a third point, pick the best two that retain the bracket
# around the minimum (as defined by HZ, eq. 29)
# b will be the upper bound, and a the lower bound
function ls_update!(ϕdϕ, alphas::Vector{T},
                    values::Vector{T}, slopes::Vector{T},
                    ia::I, ib::I, ic::I, phi_lim::T) where {I,T}
    a = alphas[ia]
    b = alphas[ib]

    zeroT = convert(T, 0)

    # Debugging (HZ, eq. 4.4):
    @assert slopes[ia] < zeroT
    @assert values[ia] <= phi_lim
    @assert slopes[ib] >= zeroT
    @assert b > a
    #
    c = alphas[ic]
    phi_c = values[ic]
    dphi_c = slopes[ic]
    #
    if c < a || c > b
        return ia, ib #, 0, 0  # it's out of the bracketing interval
    end
    #
    if dphi_c >= zeroT
        return ia, ic #, 0, 0  # replace b with a closer point
    end

    # We know dphi_c < 0. However, phi may not be monotonic between a
    # and c, so check that the value is also smaller than phi_0.  (It's
    # more dangerous to replace a than b, since we're leaving the
    # secure environment of alpha=0; that's why we didn't check this
    # above.)
    if phi_c <= phi_lim
        return ic, ib#, 0, 0  # replace a
    end

    # phi_c is bigger than phi_0, which implies that the minimum
    # lies between a and c. Find it via bisection.
    return ls_bisect!(ϕdϕ, alphas, values, slopes, ia, ic, phi_lim)
end
