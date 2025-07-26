# LM curve fitting
# curve_fit is borrowed from https://github.com/huangli712/ACFlow
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
    OnceDifferentiable(𝑓, p0::AbstractArray, 𝐹::AbstractArray)

Constructor for OnceDifferentiable struct. `𝑓` is the function, `p0` is
the inital guess, `𝐹 = 𝑓(p0)` is the cache for `𝑓`'s output.
"""
function OnceDifferentiable(𝑓, p0::AbstractArray, 𝐹::AbstractArray)
    # Backup 𝑓(x) to 𝐹.
    function ℱ!(𝐹, x)
        return copyto!(𝐹, 𝑓(x))
    end

    # Calculate jacobian for 𝑓(x), the results are stored in 𝐽.
    # The finite difference method is used.
    function 𝒥!(𝐽, x)
        rel_step = cbrt(eps(real(eltype(x))))
        abs_step = rel_step
        @inbounds for i in 1:length(x)
            xₛ = x[i]
            ϵ = max(rel_step * abs(xₛ), abs_step)
            x[i] = xₛ + ϵ
            f₂ = vec(𝑓(x))
            x[i] = xₛ - ϵ
            f₁ = vec(𝑓(x))
            𝐽[:, i] = (f₂ - f₁) ./ (2 * ϵ)
            x[i] = xₛ
        end
    end

    # Create memory space for jacobian matrix
    𝐽 = eltype(p0)(NaN) .* vec(𝐹) .* vec(p0)'

    # Call the default constructor
    return OnceDifferentiable(ℱ!, 𝒥!, 𝐹, 𝐽)
end

"""
    value(obj::OnceDifferentiable)

Return `obj.𝐹`. `obj` will not be affected.
"""
value(obj::OnceDifferentiable) = obj.𝐹

"""
    value(obj::OnceDifferentiable, 𝐹, x)

Return `𝑓(x)`. `obj` will not be affected, but `𝐹` is updated.
"""
value(obj::OnceDifferentiable, 𝐹, x) = obj.ℱ!(𝐹, x)

"""
    value!(obj::OnceDifferentiable, x)

Return `𝑓(x)`. `obj.𝐹` will be updated and returned.
"""
function value!(obj::OnceDifferentiable, x)
    obj.ℱ!(obj.𝐹, x)
    return obj.𝐹
end

"""
    jacobian(obj::OnceDifferentiable)

Return `obj.𝐽`. `obj` will not be affected.
"""
jacobian(obj::OnceDifferentiable) = obj.𝐽

"""
    jacobian(obj::OnceDifferentiable, 𝐽, x)

Return jacobian. `obj` will not be affected, but `𝐽` is updated.
"""

jacobian(obj::OnceDifferentiable, 𝐽, x) = obj.𝒥!(𝐽, x)

"""
    jacobian!(obj::OnceDifferentiable, x)

Return jacobian. `obj.𝐽` will be updated and returned.
"""
function jacobian!(obj::OnceDifferentiable, x)
    obj.𝒥!(obj.𝐽, x)
    return obj.𝐽
end

"""
    LMOptimizationResults{T,N}

It is used to save the optimization results of the `levenberg_marquardt`
algorithm.

### Members
* x₀         -> Initial guess for the solution.
* minimizer  -> Final results for the solution.
* minimum    -> Residual.
* iterations -> Number of iterations.
* xconv      -> If the convergence criterion 1 is satisfied.
* gconv      -> If the convergence criterion 2 is satisfied.
"""
struct LMOptimizationResults{T,N}
    x₀::Array{T,N}
    minimizer::Array{T,N}
    minimum::T
    iterations::Int
    xconv::Bool
    gconv::Bool
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

# === sigmoid loss function ===

# partial differentiation of sigmoid loss function
function ∂²lossϕDiv∂p²(p::Vector{T}, x::Vector{T}, y::Vector{T}) where {T<:Real}
    a, b, c, d = p
    L = length(x)
    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    s1 = s .* (1 .- s)  # s1 = s * (1 - s)
    r = a .+ b * s .- y  # 残差项

    Jaa = 2 * L
    Jbb = 2 * sum(s .^ 2)
    Jcc = 2 * b^2 * d^2 * sum(s .^ 2 .* (1 .- s) .^ 2) +
          2 * b * d^2 * sum(s1 .* (1 .- 2 * s) .* r)
    Jdd = 2 * sum(b^2 * s .^ 2 .* (1 .- s) .^ 2 .* (x .- c) .^ 2 +
                  b * (x .- c) .^ 2 .* s1 .* (1 .- 2 * s) .* r)

    Jab = 2 * sum(s)
    Jac = -2 * b * d * sum(s1)
    Jad = 2 * b * sum(s1 .* (x .- c))
    Jbc = -2 * d * sum(s1 .* (b * s .+ r))
    Jbd = 2 * sum(s1 .* (x .- c) .* (b * s .+ r))
    Jcd = -2 *
          b *
          sum(s1 .* (b * d * s1 .* (x .- c) .+ (1 .+ d * (x .- c) .* (1 .- 2 * s)) .* r))

    return [Jaa Jab Jac Jad; Jab Jbb Jbc Jbd; Jac Jbc Jcc Jcd; Jad Jbd Jcd Jdd]
end

function ∂²lossϕDiv∂p∂y(p::Vector{T}, x::Vector{T}, y::Vector{T}) where {T<:Real}
    a, b, c, d = p
    L = length(x)

    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    s1 = s .* (1 .- s)  # s1 = s * (1 - s)

    ∂²loss_∂p∂y_matrix = zeros(T, 4, L)

    ∂²loss_∂p∂y_matrix[1, :] .= -2  # ∂²loss/∂a∂y_i
    ∂²loss_∂p∂y_matrix[2, :] = -2 * s  # ∂²loss/∂b∂y_i
    ∂²loss_∂p∂y_matrix[3, :] = 2 * b * d * s1  # ∂²loss/∂c∂y_i
    ∂²loss_∂p∂y_matrix[4, :] = -2 * b * s1 .* (x .- c)  # ∂²loss/∂d∂y_i

    return ∂²loss_∂p∂y_matrix
end
#=
function ∂lossϕDiv∂y(p, x, y)
    a, b, c, d = p
    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    return -2 * (a .+ b * s .- y)
end

function lossϕ(p, x, y)
    a, b, c, d=p
    s=1 ./ (1 .+ exp.(-d*(x .- c)))
    r=a .+ b*s - y
    return sum(r .^ 2)
end

function ∂lossϕDiv∂p(p, x, y)
    a, b, c, d=p
    s=1 ./ (1 .+ exp.(-d*(x .- c)))
    r=a .+ b*s - y
    Ja=2*sum(r)
    Jb=2*sum(s .* r)
    Jc=-2*b*d*sum(s .* (1 .- s) .* r)
    Jd=2*b*sum(s .* (1 .- s) .* (x .- c) .* r)
    return [Ja, Jb, Jc, Jd]
end
=#
