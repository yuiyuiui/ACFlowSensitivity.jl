"""
    curve_fit(model, x, y, p0)

Fit data to a non-linear `model`. `p0` is an initial model parameter guess.
The return object is a composite type (`LsqFitResult`).

See also: [`LsqFitResult`](@ref).
"""
function curve_fit(model, x::AbstractArray, y::AbstractArray, p0::AbstractArray)
    f = (p) -> model(x, p) - y
    r = f(p0)
    R = OnceDifferentiable(f, p0, r)
    OR = levenberg_marquardt(R, p0)
    p = OR.minimizer
    conv = OR.xconv || OR.gconv
    return LsqFitResult(p, value!(R, p), jacobian!(R, p), conv)
end

"""
    OnceDifferentiable

Mutable struct. It is used for objectives and solvers where the gradient
is available/exists.

### Members
* ℱ! -> Objective. It is actually a function call and return objective.
* 𝒥! -> It is a function call as well and returns jacobian of objective.
* 𝐹  -> Cache for ℱ! output.
* 𝐽  -> Cache for 𝒥! output.
"""
mutable struct OnceDifferentiable
    ℱ!::Any
    𝒥!::Any
    𝐹::Any
    𝐽::Any
end

"""
    levenberg_marquardt(df::OnceDifferentiable, x₀::AbstractVector{T})

Returns the argmin over x of `sum(f(x).^2)` using the Levenberg-Marquardt
algorithm. The function `f` is encoded in `df`. `x₀` is an initial guess
for the solution.

See also: [`OnceDifferentiable`](@ref).
"""
function levenberg_marquardt(df::OnceDifferentiable, x₀::AbstractVector{T} where {T})
    # Some predefined constants
    min_diagonal = 1e-6 # lower bound on values of diagonal matrix
    #
    x_tol = 1e-08 # search tolerance in x
    g_tol = 1e-12 # search tolerance in gradient
    maxIter = 1000  # maximum number of iterations
    #
    Λₘ = 1e+16 # minimum trust region radius
    λₘ = 1e-16 # maximum trust region radius
    λ = eltype(x₀)(10) # (inverse of) initial trust region radius
    λᵢ = 10.0  # λ is multiplied by this factor after step below min quality
    λᵣ = 0.10  # λ is multiplied by this factor after good quality steps
    #
    min_step_quality = 1e-3 # for steps below this quality, the trust region is shrinked
    good_step_quality = 0.75 # for steps above this quality, the trust region is expanded

    # First evaluation
    # Both df.𝐹 and df.𝐽 are updated.
    # And 𝐹 and 𝐽 become aliases of df.𝐹 and df.𝐽, respectively.
    value!(df, x₀)
    jacobian!(df, x₀)
    𝐹 = value(df)
    𝐽 = jacobian(df)

    # Setup convergence criteria
    converged = false
    xconv = false
    gconv = false
    iter = 0

    # Calculate 𝑓(x₀) and initial residual
    x = copy(x₀)
    trial_f = similar(𝐹)
    C_resid = sum(abs2, 𝐹)

    # Create buffers
    𝐽ᵀ𝐽 = diagm(x)
    𝐽δx = similar(𝐹)

    # Main iteration
    while (~converged && iter < maxIter)
        # Update jacobian 𝐽 for new x
        jacobian!(df, x)

        # Solve the equation: [𝐽ᵀ𝐽 + λ diag(𝐽ᵀ𝐽)] δ = 𝐽ᵀ𝐹
        # What we want to get is δ.
        mul!(𝐽ᵀ𝐽, 𝐽', 𝐽)
        #
        𝐷ᵀ𝐷 = diag(𝐽ᵀ𝐽)
        replace!(x -> x ≤ min_diagonal ? min_diagonal : x, 𝐷ᵀ𝐷)
        #
        @simd for i in eachindex(𝐷ᵀ𝐷)
            @inbounds 𝐽ᵀ𝐽[i, i] += λ * 𝐷ᵀ𝐷[i]
        end
        #
        δx = - 𝐽ᵀ𝐽 \ (𝐽' * 𝐹)

        # If the linear assumption is valid, the new residual is predicted.
        mul!(𝐽δx, 𝐽, δx)
        𝐽δx .= 𝐽δx .+ 𝐹
        P_resid = sum(abs2, 𝐽δx)

        # Try to calculate new x, and then 𝐹 ≡ 𝑓(x), and then the residual.
        xnew = x + δx
        value(df, trial_f, xnew)
        T_resid = sum(abs2, trial_f)

        # Step quality = residual change / predicted residual change
        ρ = (T_resid - C_resid) / (P_resid - C_resid)
        if ρ > min_step_quality
            # Update x, 𝑓(x), and residual.
            x .= xnew
            value!(df, x)
            C_resid = T_resid

            # Increase trust region radius
            if ρ > good_step_quality
                λ = max(λᵣ * λ, λₘ)
            end
        else
            # Decrease trust region radius
            λ = min(λᵢ * λ, Λₘ)
        end

        # Increase the iteration
        iter += 1

        # Check convergence criteria:
        # 1. Small gradient: norm(𝐽ᵀ * 𝐹, Inf) < g_tol
        if norm(𝐽' * 𝐹, Inf) < g_tol
            gconv = true
        end
        # 2. Small step size: norm(δx) < x_tol
        if norm(δx) < x_tol * (x_tol + norm(x))
            xconv = true
        end
        # 3. Calculate converged
        converged = gconv | xconv
    end

    # Return the results
    return LMOptimizationResults(x₀,      # x₀
                                 x,       # minimizer
                                 C_resid, # residual
                                 iter,    # iterations
                                 xconv,   # xconv
                                 gconv)
end

"""
    LsqFitResult

It encapsulates the results for curve fitting.

### Members
* param     -> Fitted results, i.e, the fitting parameters.
* resid     -> Residuals.
* jacobian  -> Jacobian matrix.
* converged -> If the curve-fitting algorithm is converged.
"""
struct LsqFitResult
    param::Any
    resid::Any
    jacobian::Any
    converged::Any
end
