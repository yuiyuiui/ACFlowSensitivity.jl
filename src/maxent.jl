#
# Project : Gardenia
# Source  : maxent.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/09/30
#

#=
### *Customized Structs* : *MaxEnt Solver*
=#

"""
    MaxEntContext

Mutable struct. It is used within the MaxEnt solver only.

### Members
* Gᵥ     -> Input data for correlator.
* σ⁻²     -> Actually 1.0 / σ².
* grid   -> Grid for input data.
* mesh   -> Mesh for output spectrum.
* model  -> Default model function.
* K      -> Kernel matrix.
* U      -> Matrix from singular value decomposition.
* Σ      -> Diagonal matrix from singular value decomposition.
* V      -> Matrix from singular value decomposition.
* stype  -> Type of entropy.
* hess   -> Hessian matrix.
* Vₛ     -> Matrix from singular value decomposition.
* W₂     -> Precomputed array.
* W₃     -> Precomputed array.
* Bₘ     -> Precomputed array.
"""
mutable struct MaxEntContext{T<:Real}
    Gᵥ::Vector{T}
    σ⁻²::Vector{T}
    grid::Vector{T}
    mesh::Vector{T}  # mesh
    δ::Vector{T}     # mesh_weight
    model::Vector{T}
    kernel::Array{T,2}
    U::Array{T,2}
    Σ::Diagonal{T,Vector{T}}
    V::Array{T,2}
    stype::Stype
    hess::Array{T,2}
    Vₛ::Array{T,2}
    W₂::Array{T,2}
    W₃::Array{T,3}
    Bₘ::Vector{T}
end

#=
### *Global Drivers*
=#

"""
    solve(GFV::Vector{Complex{T}}, ctx::CtxData{T},
          alg::MaxEnt) where {T<:Real}

Solve the analytic continuation problem by the maximum entropy method. It
is the driver for the MaxEnt solver.

If the input correlators are bosonic, this solver will return A(ω) / ω
via `Aout`, instead of A(ω). At this time, `Aout` is not compatible with
`Gout`. If the input correlators are fermionic, this solver will return
A(ω) in `Aout`. Now it is compatible with `Gout`. These behaviors are just
similar to the StochAC, StochSK, and StochOM solvers.

It seems that the MaxEnt solver is hard to create δ-like spectra.

### Arguments
* GFV -> Vector of complex numbers, containing the input data.
* ctx -> CtxData struct, containing the context data.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* mesh -> Real frequency mesh, ω.
* Aout -> Spectral function, A(ω).
"""
function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T},
               alg::MaxEnt) where {T<:Real}
    println("[ MaxEnt ]")
    mec = init(GFV, ctx, alg)
    _, sol = run!(mec, alg)
    Aout = sol[:A]
    if ctx.spt isa Cont
        return Aout
    elseif ctx.spt isa Delta
        p = ctx.mesh.mesh[find_peaks(ctx.mesh.mesh, Aout, ctx.fp_mp; wind=ctx.fp_ww)]
        γ = pG2γ(p, GFV, ctx.iwn)
        return Aout, (p, γ)
    else
        error("Unsupported spectral function type")
    end
end

"""
    init(GFV::Vector{Complex{T}}, ctx::CtxData{T},
         alg::MaxEnt) where {T<:Real}

Initialize the MaxEnt solver and return a MaxEntContext struct.

### Arguments
* GFV -> Vector of complex numbers, containing the input data.
* ctx -> CtxData struct, containing the context data.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* mec -> A MaxEntContext struct.
"""
function init(GFV::Vector{Complex{T}}, ctx::CtxData{T},
              alg::MaxEnt) where {T<:Real}
    # Prepera input data
    G, K, _, U, S, V = SingularSpace(GFV, ctx.wn, ctx.mesh.mesh)
    σ⁻² = (1 ./ T(ctx.σ))^2 * ones(T, length(G))
    model = make_model(alg.model_type, ctx)

    # Prepare some essential intermediate variables
    Vₛ, W₂, W₃, Bₘ, hess = precompute(GFV, σ⁻², ctx.mesh, ctx.wn, model)
    println("Precompute key coefficients")

    return MaxEntContext(G, σ⁻², ctx.wn, ctx.mesh.mesh, ctx.mesh.weight, model,
                         K, U, Diagonal(S), V, alg.stype, hess, Vₛ, W₂, W₃, Bₘ)
end

"""
    run!(mec::MaxEntContext, alg::MaxEnt)

Perform maximum entropy simulation with different algorithms. Now it
supports the `historic`, `classic`, `bryan`, and `chi2kink` algorithms.

### Arguments
* mec -> A MaxEntContext struct.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* svec -> A vector of dictionaries. It contains the intermediate solutions.
* sol -> Dictionary. It contains the final solution.
"""
function run!(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    method = alg.method

    # Note that the Bayesian Reconstruction entropy is compatible with
    # all the four algorithms so far.
    if mec.stype isa BR
        println("Bayesian Reconstruction entropy is used!")
    else
        println("Shannon–Jaynes entropy is used!")
    end

    method == "historic" && return historic(mec, alg)
    method == "classic" && return classic(mec, alg)
    method == "bryan" && return bryan(mec, alg)
    method == "chi2kink" && return chi2kink(mec, alg)
end

#=
### *Core Algorithms*
=#

"""
    historic(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}

Apply the historic algorithm to solve the analytic continuation problem.
It choose α in a way that χ² ≈ N.

For the historic algorithm, `alpha` is usually 10⁶, and `ratio` is 10.0.
It is compatible with the Bayesian Reconstruction entropy.

### Arguments
* mec -> A MaxEntContext struct.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* svec -> A vector of dictionaries. It contains the intermediate solutions.
* sol -> Dictionary. It contains the final solution.

See also: [`MaxEntContext`](@ref).
"""
function historic(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    function root_fun(_α, _u)
        res = optimizer(mec, _α, _u, use_bayes, alg)
        @. _u = res[:u]
        return length(mec.σ⁻²) / res[:χ²] - T(1)
    end

    println("Apply historic algorithm to determine optimized α")

    use_bayes = false
    alpha = T(alg.alpha)
    ratio = T(alg.ratio)
    n_svd = length(mec.Bₘ)

    u_vec = zeros(T, n_svd)
    s_vec = []

    conv = T(0)
    while conv < T(1)
        sol = optimizer(mec, alpha, u_vec, use_bayes, alg)
        push!(s_vec, sol)
        alpha = alpha / ratio
        conv = length(mec.σ⁻²) / sol[:χ²]
    end

    u_vec = s_vec[end-1][:u]
    alpha = s_vec[end][:α]
    α_opt = secant(root_fun, alpha, u_vec)

    sol = optimizer(mec, α_opt, u_vec, use_bayes, alg)
    println("Optimized α : $α_opt log10(α) : $(log10(α_opt))")

    return s_vec, sol
end

"""
    classic(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}

Apply the classic algorithm to solve the analytic continuation problem.

Classic algorithm uses Bayes statistics to approximately determine the
most probable value of α. We always start at a large value of α, where
the optimization yields basically the default model, therefore `u_vec`
is only a few steps away from 0 (= default model). And then we gradually
decrease α, step by step moving away from the default model towards data
fitting. Using `u_vec` as start for the next (smaller) α brings a great
speedup into this procedure.

For the classic algorithm, `alpha` is usually 10⁶, and `ratio` is 10.0.
It is incompatible with the Bayesian Reconstruction entropy.

### Arguments
* mec -> A MaxEntContext struct.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* svec -> A vector of dictionaries. It contains the intermediate solutions.
* sol -> Dictionary. It contains the final solution.

See also: [`MaxEntContext`](@ref).
"""
function classic(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    function root_fun(_α, _u)
        res = optimizer(mec, _α, _u, use_bayes, alg)
        @. _u = res[:u]
        return res[:conv] - T(1)
    end

    println("Apply classic algorithm to determine optimized α")

    use_bayes = true
    alpha = T(alg.alpha)
    ratio = T(alg.ratio)
    n_svd = length(mec.Bₘ)

    u_vec = zeros(T, n_svd)
    s_vec = []

    conv = T(0)
    while conv < T(1)
        sol = optimizer(mec, alpha, u_vec, use_bayes, alg)
        push!(s_vec, sol)
        alpha = alpha / ratio
        @. u_vec = sol[:u]
        conv = sol[:conv]
    end

    c_vec = [x[:conv] for x in s_vec]
    α_vec = [x[:α] for x in s_vec]
    exp_opt = log10(α_vec[end] / α_vec[end-1])
    exp_opt = exp_opt / log10(c_vec[end] / c_vec[end-1])
    exp_opt = log10(α_vec[end-1]) - log10(c_vec[end-1]) * exp_opt

    # Starting from the predicted value of α, and starting optimization
    # at the solution for the next-lowest α, we find the optimal α by
    # secant root finding method.
    u_vec = s_vec[end-1][:u]
    alpha = T(10) ^ exp_opt
    α_opt = secant(root_fun, alpha, u_vec)

    sol = optimizer(mec, α_opt, u_vec, use_bayes, alg)
    println("Optimized α : $α_opt log10(α) : $(log10(α_opt))")

    return s_vec, sol
end

"""
    bryan(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}

Apply the bryan algorithm to solve the analytic continuation problem.

Bryan's maxent calculates an average of spectral functions, weighted by
their Bayesian probability.

For the bryan algorithm, `alpha` is usually 500, and `ratio` is 1.1.
It is incompatible with the Bayesian Reconstruction entropy.

### Arguments
* mec -> A MaxEntContext struct.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* svec -> A vector of dictionaries. It contains the intermediate solutions.
* sol -> Dictionary. It contains the final solution.

See also: [`MaxEntContext`](@ref).
"""
function bryan(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    println("Apply bryan algorithm to determine optimized α")

    use_bayes = true
    alpha = T(alg.alpha)
    ratio = T(alg.ratio)
    n_svd = length(mec.Bₘ)
    nmesh = length(mec.mesh)

    u_vec = zeros(T, n_svd)
    s_vec = []

    maxprob = T(0)
    while true
        sol = optimizer(mec, alpha, u_vec, use_bayes, alg)
        push!(s_vec, sol)
        alpha = alpha / ratio
        alpha == 0 && break
        @. u_vec = sol[:u]
        prob = sol[:prob]
        if prob > maxprob
            maxprob = prob
        elseif prob < T(0.01) * maxprob
            break
        end
    end

    α_vec = map(x->x[:α], s_vec)
    p_vec = map(x->x[:prob], s_vec)
    p_vec = -p_vec ./ trapz(α_vec, p_vec)
    A_vec = map(x->x[:A], s_vec)

    nprob = length(p_vec)
    A_opt = zeros(T, nmesh)
    spectra = zeros(T, nmesh, nprob)
    for i in 1:nprob
        spectra[:, i] = A_vec[i] * p_vec[i]
    end
    for j in 1:nmesh
        A_opt[j] = -trapz(α_vec, spectra[j, :])
    end

    sol = Dict(:A => A_opt)

    return s_vec, sol
end

"""
    chi2kink(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}

Apply the chi2kink algorithm to solve the analytic continuation problem.

We start with an optimization at a large value of α, where we should get
only the default model. And then, α is decreased step-by-step, until the
minimal value of α is reached. Then, we fit a function

`ϕ(x; a, b, c, d) = a + b / [1 + exp(-d*(x-c))]`,

from which the optimal α is determined by

`x_opt = c - fit_position / d`,

and

`alpha_opt = 10^x_opt`.

For the chi2kink algorithm, `alpha` is usually 10⁹, `ratio` is 10.0, the
number of alpha parameters is 12. It is compatible with the Bayesian
Reconstruction entropy.

### Arguments
* mec -> A MaxEntContext struct.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* svec -> A vector of dictionaries. It contains the intermediate solutions.
* sol -> Dictionary. It contains the final solution.

See also: [`MaxEntContext`](@ref).
"""
function chi2kink(mec::MaxEntContext{T}, alg::MaxEnt) where {T<:Real}
    function fitfun(x, p)
        return @. p[1] + p[2] / (T(1) + exp(-p[4] * (x - p[3])))
    end

    println("Apply chi2kink algorithm to determine optimized α")

    use_bayes = false
    alpha = T(alg.alpha)
    ratio = T(alg.ratio)
    nalph = T(alg.nalph)
    α_end = alpha / (ratio^nalph)
    n_svd = length(mec.Bₘ)

    u_vec = zeros(T, n_svd)
    s_vec = []
    χ_vec = []
    α_vec = []

    while true
        sol = optimizer(mec, alpha, u_vec, use_bayes, alg)
        push!(s_vec, sol)
        push!(α_vec, alpha)
        push!(χ_vec, sol[:χ²])
        @. u_vec = sol[:u]
        alpha = alpha / ratio
        if alpha < α_end
            break
        end
    end

    good = isfinite.(χ_vec)
    guess = [T(0), T(5), T(2), T(0)]
    fit = curve_fit(fitfun, log10.(α_vec[good]), log10.(χ_vec[good]), guess)
    _, _, c, d = fit.param

    # `fit_pos` is a control parameter for under/overfitting.
    # Good values are usually between 2 and 2.5. Smaller values usually
    # lead to underfitting, which is sometimes desirable. Larger values
    # lead to overfitting, which should be avoided.
    fit_pos = T(2.5)
    α_opt = c - fit_pos / d
    close = argmin(abs.(log10.(α_vec) .- α_opt))
    u_vec = s_vec[close][:u]
    α_opt = T(10) ^ α_opt

    sol = optimizer(mec, α_opt, u_vec, use_bayes, alg)
    println("Optimized α : $α_opt log10(α) : $(log10(α_opt))")

    return s_vec, sol
end

"""
    optimizer(
        mec::MaxEntContext,
        α::T,
        us::Vector{T},
        use_bayes::Bool,
        alg::MaxEnt
    ) where {T<:Real}

Optimization of maxent functional for a given value of `α`. Since a priori
the best value of `α` is unknown, this function has to be called several
times in order to find a good value.

`α` means a weight factor of the entropy. `us` is a vector in singular
space. It is used as a starting value for the optimization. For the very
first optimization, done at large α, we use zeros, which corresponds to
the default model. Then we use the result of the previous optimization
as a starting value. `use_bayes` determines whether to use the Bayesian
inference parameters for `α`.

This function will return a dictionary object that holds the results of
the optimization, e.g. spectral function, χ² deviation.

### Arguments
* mec -> A MaxEntContext struct.
* α -> See above explanations.
* us -> See above explanations.
* use_bayes -> See above explanations.
* alg -> MaxEnt struct, containing the algorithm.

### Returns
* dict -> A dictionary, the solution to analytic continuation problem.
"""
function optimizer(mec::MaxEntContext{T},
                   α::T,
                   us::Vector{T},
                   use_bayes::Bool,
                   alg::MaxEnt) where {T<:Real}
    blur = T(alg.blur)
    offdiag = alg.offdiag
    stype = alg.stype

    if offdiag
        solution, call = newton(f_and_J_od, us, mec, α, stype)
        u = copy(solution)
        A = svd_to_real_od(mec, solution, stype)
        S = calc_entropy_od(mec, A, stype)
    else
        solution, call = newton(f_and_J, us, mec, α, stype)
        u = copy(solution)
        A = svd_to_real(mec, solution, stype)
        S = calc_entropy(mec, A, u, stype)
    end

    χ² = calc_chi2(mec, A)
    norm = trapz(mec.mesh, A)

    dict = Dict{Symbol,Any}(:u => u,
                            :α => α,
                            :S => S,
                            :χ² => χ²,
                            :norm => norm,
                            :Q => α * S - T(1//2) * χ²,
                            :Araw => deepcopy(A))

    if use_bayes
        if offdiag
            ng, tr, conv, prob = calc_bayes_od(mec, A, S, χ², α, stype)
        else
            ng, tr, conv, prob = calc_bayes(mec, A, S, χ², α, stype)
        end
        dict[:ngood] = ng
        dict[:trace] = tr
        dict[:conv] = conv
        dict[:prob] = prob
    end

    if blur > T(0)
        make_blur(mec.mesh, A, blur)
    end
    dict[:A] = A
    return dict
end

#=
### *Service Functions*
=#

#=
*Remarks* :

Try to calculate some key variables by using the Einstein summation trick.

```math
\begin{equation}
B_m = \sum^{N}_{n = 1} \frac{1}{\sigma^2_n} \xi_m U_{nm} G_n,
\end{equation}
```

```math
\begin{equation}
W_{ml} = \sum_{pn} \frac{1}{\sigma^2_n}
    U_{nm}\xi_m U_{np} \xi_p V_{lp} \Delta_l D_l,
\end{equation}
```

```math
\begin{equation}
W_{mli} = W_{ml} V_{li}.
\end{equation}
```

Note that these variables do not depend on the spectral function
``A(\omega)``, so they could be computed at advance to improve the
computational efficiency.

---

The `hessian matrix` is also calculated here.

```math
\begin{equation}
L = \frac{1}{2} \chi^2,
\end{equation}
```

```math
\begin{equation}
\frac{\partial^2 L}{\partial A_i \partial A_j} =
\sum_n \frac{K_{ni} \Delta_i K_{nj} \Delta_j}{\sigma^2_n}.
\end{equation}
```
=#

"""
    precompute(
        GFV::Vector{Complex{T}},
        σ⁻²::Vector{T},
        am::Mesh{T},
        grid::Vector{T},
        D::Vector{T},
    ) where {T<:Real}

Precompute some key coefficients. Here `GFV` and `σ⁻²` are input data, `am`
is the mesh for spectrum, `grid` is the mesh for frequency, `D` is the
default model.

### Arguments
* GFV -> Input correlator.
* σ⁻² -> Error bar for input correlator.
* am -> See above explanations.
* grid -> See above explanations.
* D -> See above explanations.

### Returns
* V -> An orthogonal matrix from singular value decomposition of kernel.
* W₂ -> The Wₘₗ matrix.
* W₃ -> The Wₘₗᵢ tensor.
* Bₘ -> The Bₘ vector.
* hess -> The Hessian matrix.
"""
function precompute(GFV::Vector{Complex{T}},
                    σ⁻²::Vector{T},
                    am::Mesh{T},
                    grid::Vector{T},
                    D::Vector{T}) where {T<:Real}
    # Create singular value space
    Gᵥ, K, n_svd, U, S, V = SingularSpace(GFV, grid, am.mesh)

    # Evaluate sizes of the arrays
    nmesh = length(am.mesh)

    # Allocate memories
    W₂ = zeros(T, n_svd, nmesh)
    W₃ = zeros(T, n_svd, n_svd, nmesh)
    Bₘ = zeros(T, n_svd)
    hess = zeros(T, nmesh, nmesh)

    # Get weight of the mesh, Δωₗ.
    Δ = am.weight

    # Compute Wₘₗ
    @einsum W₂[m, l] = σ⁻²[k] * U[k, m] * S[m] * U[k, n] * S[n] * V[l, n] * Δ[l] * D[l]

    # Compute Wₘₗᵢ
    @einsum W₃[m, k, l] = W₂[m, l] * V[l, k]

    # Compute Bₘ
    @einsum Bₘ[m] = S[m] * U[k, m] * σ⁻²[k] * Gᵥ[k]

    # Compute the Hessian matrix
    @einsum hess[i, j] = Δ[i] * Δ[j] * K[k, i] * K[k, j] * σ⁻²[k]
    hess = (hess + hess') / 2

    return V, W₂, W₃, Bₘ, hess
end

#=
*Remarks* :

For Shannon-Jaynes entropy,

```math
\begin{equation}
w_l = \exp \left(\sum_m V_{lm} u_m\right).
\end{equation}
```

```math
\begin{equation}
f_m = \alpha u_m + \sum_l W_{ml} w_l - B_m.
\end{equation}
```

```math
\begin{equation}
J_{mi} = \alpha \delta_{mi} + \sum_l W_{mli} w_l.
\end{equation}
```

---

For Bayesian Reconstruction entropy,

```math
\begin{equation}
w_l = \frac{1}{1 - D_l \sum_m V_{lm} u_m}.
\end{equation}
```

```math
\begin{equation}
f_m = \alpha u_m + \sum_l W_{ml} w_l - B_m.
\end{equation}
```

```math
\begin{equation}
J_{mi} = \alpha \delta_{mi} + \sum_l W_{mli} D_l w_l w_l.
\end{equation}
```
=#

"""
    f_and_J(u::Vector{T}, mec::MaxEntContext, α::T, stype::Stype) where {T<:Real}

This function evaluates the function whose root we want to find. Here
`u` is a singular space vector that parametrizes the spectral function,
and `α` is a (positive) weight factor of the entropy.

It returns `f`, value of the function whose zero we want to find, and
`J`, jacobian at the current position.

### Arguments
See above explanations.

### Returns
See above explanations.

See also: [`f_and_J_od`](@ref).
"""
function f_and_J(u::Vector{T}, mec::MaxEntContext{T}, α::T, stype::Stype) where {T<:Real}
    n_svd = length(mec.Bₘ)
    J = diagm([α for i in 1:n_svd])

    # For Shannon–Jaynes entropy
    if mec.stype isa SJ
        w = exp.(mec.Vₛ * u)
        #
        for j in 1:n_svd
            for i in 1:n_svd
                J[i, j] = J[i, j] + dot(mec.W₃[i, j, :], w)
            end
        end
        #
        f = α * u + mec.W₂ * w - mec.Bₘ
        # For Bayesian Reconstruction entropy
    else
        w = mec.Vₛ * u
        w₁ = 1 ./ (1 .- mec.model .* w)
        w₂ = w₁ .* w₁ .* mec.model
        #
        for j in 1:n_svd
            for i in 1:n_svd
                J[i, j] = J[i, j] + dot(mec.W₃[i, j, :], w₂)
            end
        end
        #
        f = α * u + mec.W₂ * w₁ - mec.Bₘ
    end

    return f, J
end

#=
*Remarks* :

For Shannon-Jaynes entropy,

```math
\begin{equation}
w_l = \exp \left(\sum_m V_{lm} u_m\right).
\end{equation}
```

```math
\begin{equation}
f_m = \alpha u_m +
      \sum_l W_{ml}\left(w_l - \frac{1}{w_l}\right) - B_m.
\end{equation}
```

```math
\begin{equation}
J_{mi} = \alpha \delta_{mi} +
         \sum_{l} W_{mli} \left(w_l + \frac{1}{w_l}\right).
\end{equation}
```

---

For Bayesian Reconstruction entropy,

```math
\begin{equation}
w^+_l = \frac{1}{ 1 - D_l \sum_m V_{lm} u_m}.
\end{equation}
```

```math
\begin{equation}
w^-_l = \frac{1}{ 1 + D_l \sum_m V_{lm} u_m}.
\end{equation}
```

```math
\begin{equation}
f_m = \alpha u_m + \sum_l W_{ml} (w^+_l - w^-_l) - B_m.
\end{equation}
```

```math
\begin{equation}
J_{mi} = \alpha \delta_{mi} + \sum_l W_{mli} D_l (w^+_l w^+_l + w^-_l w^-_l).
\end{equation}
```
=#

"""
    f_and_J_od(u::Vector{T}, mec::MaxEntContext, α::T, stype::Stype) where {T<:Real}

This function evaluates the function whose root we want to find. Here
`u` is a singular space vector that parametrizes the spectral function,
and `α` is a (positive) weight factor of the entropy.

It returns `f`, value of the function whose zero we want to find, and
`J`, jacobian at the current position.

This function is similar to `f_and_J`, but for offdiagonal elements.

### Arguments
See above explanations.

### Returns
See above explanations.

See also: [`f_and_J`](@ref).
"""
function f_and_J_od(u::Vector{T}, mec::MaxEntContext{T}, α::T,
                    stype::Stype) where {T<:Real}
    n_svd = length(mec.Bₘ)
    J = diagm([α for i in 1:n_svd])

    # For Shannon–Jaynes entropy
    if mec.stype isa SJ
        w = exp.(mec.Vₛ * u)
        #
        a⁺ = 1 .* w
        a⁻ = 1 ./ w
        a₁ = a⁺ - a⁻
        a₂ = a⁺ + a⁻
        #
        for j in 1:n_svd
            for i in 1:n_svd
                J[i, j] = J[i, j] + dot(mec.W₃[i, j, :], a₂)
            end
        end
        #
        f = α * u + mec.W₂ * a₁ - mec.Bₘ
        # For Bayesian Reconstruction entropy
    else
        w = mec.Vₛ * u
        #
        a⁺ = 1 ./ (1 .- mec.model .* w)
        a⁻ = 1 ./ (1 .+ mec.model .* w)
        a₁ = a⁺ - a⁻
        a₂ = (a⁺ .* a⁺ + a⁻ .* a⁻) .* mec.model
        #
        for j in 1:n_svd
            for i in 1:n_svd
                J[i, j] = J[i, j] + dot(mec.W₃[i, j, :], a₂)
            end
        end
        #
        f = α * u + mec.W₂ * a₁ - mec.Bₘ
    end

    return f, J
end

#=
*Remarks* :

For Shannon-Jaynes entropy,

```math
\begin{equation}
A_l = D_l \exp \left(\sum_m V_{lm} u_m\right).
\end{equation}
```

---

For Bayesian Reconstruction entropy,

```math
\begin{equation}
A_l = \frac{D_l}{ 1 - D_l \sum_m V_{lm} u_m}.
\end{equation}
```
=#

"""
    svd_to_real(mec::MaxEntContext, u::Vector{T}, stype::Stype) where {T<:Real}

Go from singular value space to real space. It will transform the singular
space vector `u` into real-frequency space (to get the spectral function)
by `A(ω) = D(ω) eⱽᵘ`, where `D(ω)` is the default model, `V` is the matrix
from the singular value decomposition. The argument `u` means a singular
space vector that parametrizes the spectral function.

### Arguments
See above explanations.

### Returns
See above explanations.

See also: [`svd_to_real_od`](@ref).
"""
function svd_to_real(mec::MaxEntContext{T}, u::Vector{T}, stype::Stype) where {T<:Real}
    #
    # For Shannon–Jaynes entropy
    if mec.stype isa SJ
        w = exp.(mec.Vₛ * u)
        return mec.model .* w
        # For Bayesian Reconstruction entropy
    else
        w = mec.Vₛ * u
        return mec.model ./ (1 .- mec.model .* w)
    end
end

#=
*Remarks* :

For Shannon-Jaynes entropy,

```math
\begin{equation}
A_l = D_l \exp \left(\sum_m V_{lm} u_m\right) -
      D_l \exp \left(-\sum_m V_{lm} u_m\right).
\end{equation}
```

---

For Bayesian Reconstruction entropy,

```math
\begin{equation}
w^+_l = \frac{1}{ 1 - D_l \sum_m V_{lm} u_m}.
\end{equation}
```

```math
\begin{equation}
w^-_l = \frac{1}{ 1 + D_l \sum_m V_{lm} u_m}.
\end{equation}
```

```math
\begin{equation}
A_l = D_l (w^+_l - w^-_l).
\end{equation}
```
=#

"""
    svd_to_real_od(mec::MaxEntContext, u::Vector{T}, stype::Stype) where {T<:Real}

Go from singular value space to real space. It will transform the singular
space vector `u` into real-frequency space in the case of an offdiagonal
element. It will return the spectral function.

### Arguments
* mec -> A MaxEntContext struct.
* u -> A singular space vector that parametrizes the spectral function.

### Returns
See above explanations.

See also: [`svd_to_real`](@ref).
"""
function svd_to_real_od(mec::MaxEntContext{T}, u::Vector{T}, stype::Stype) where {T<:Real}
    #
    # For Shannon–Jaynes entropy
    if mec.stype isa SJ
        w = exp.(mec.Vₛ * u)
        w⁺ = w
        w⁻ = 1 ./ w
        return mec.model .* (w⁺ .- w⁻)
        # For Bayesian Reconstruction entropy
    else
        w = mec.Vₛ * u
        w⁺ = 1 ./ (1 .- mec.model .* w)
        w⁻ = 1 ./ (1 .+ mec.model .* w)
        return mec.model .* (w⁺ .- w⁻)
    end
end

#=
*Remarks* :

Shannon–Jaynes entropy

```math
\begin{equation}
S[A] = \int^{\infty}_0 d\omega
\left[
    A - m -
    A \log{\left(\frac{A}{m}\right)}
\right],
\end{equation}
```

```math
\begin{equation}
S[A^{+},A^{-}] = \int^{+\infty}_0 d\omega
\left[
    \sqrt{A^2 + 4m^2} - 2m -
    A\log{\left(\frac{\sqrt{A^2 + 4m^2} + A}{2m}\right)}
\right].
\end{equation}
```

---

Bayesian Reconstruction entropy

```math
\begin{equation}
S[A] = \int^{\infty}_0 d\omega
\left[
    1 - \frac{A}{m} + \log{\left(\frac{A}{m}\right)}
\right],
\end{equation}
```

```math
\begin{equation}
S[A^{+},A^{-}] = \int^{+\infty}_0 d\omega
\left[
    2 - \frac{\sqrt{A^2 + m^2} + m}{m} +
    \log{\left(\frac{\sqrt{A^2 + m^2} + m}{2m}\right)}
\right].
\end{equation}
```
=#

"""
    calc_entropy(mec::MaxEntContext, A::Vector{T}, u::Vector{T}, stype::Stype) where {T<:Real}

It computes entropy for positive definite spectral function. Here the
arguments `A` means spectral function and `u` means a singular space
vector that parametrizes the spectral function.

### Arguments
See above explanations.

### Returns
* S -> Entropy.

See also: [`calc_entropy_od`](@ref).
"""
function calc_entropy(mec::MaxEntContext{T}, A::Vector{T}, u::Vector{T},
                      stype::Stype) where {T<:Real}
    #
    # For Shannon–Jaynes entropy
    if mec.stype isa SJ
        f = A - mec.model - A .* (mec.Vₛ * u)
        # For Bayesian Reconstruction entropy
    else
        𝑅 = A ./ mec.model
        #
        if any(x -> x < T(0), 𝑅)
            @info "Negative spectrum occurs!"
            @info "The results might be questionable."
            @info "Perhaps you should switch to the Shannon–Jaynes entropy."
            f = 1 .- 𝑅 + log.(abs.(𝑅))
        else
            f = 1 .- 𝑅 + log.(𝑅)
        end
    end
    #
    return trapz(mec.mesh, f)
end

"""
    calc_entropy_od(mec::MaxEntContext, A::Vector{T}, stype::Stype) where {T<:Real}

It compute *positive-negative entropy* for spectral function with norm 0.
Here the argument `A` means spectral function.

### Arguments
See above explanations.

### Returns
* S -> Entropy.

See also: [`calc_entropy`](@ref).
"""
function calc_entropy_od(mec::MaxEntContext{T}, A::Vector{T}, stype::Stype) where {T<:Real}
    #
    # For Shannon–Jaynes entropy
    if mec.stype isa SJ
        root = sqrt.(A .^ 2 + 4 .* mec.model .* mec.model)
        f = root - 2 .* mec.model
        f = f - A .* log.((root + A) ./ (2 .* mec.model))
        # For Bayesian Reconstruction entropy
    else
        root = sqrt.(A .^ 2 + mec.model .^ 2) + mec.model
        f = 2 .- (root ./ mec.model) + log.(root ./ (2 .* mec.model))
    end
    #
    return trapz(mec.mesh, f)
end

#=
*Remarks* :

**Posterior distribution of ``\alpha``**

Because
```math
\begin{equation}
\text{Pr}[\alpha | \bar{G}] =
\text{Pr}[\alpha] \frac{e^Q}{Z_L Z_S}
\frac{(2\pi)^{N/2}}{\sqrt{\det{[\alpha I + \Lambda]}}},
\end{equation}
```

so
```math
\begin{equation}
\log \text{Pr}[\alpha | \bar{G}] =
\text{constant} + \log \text{Pr} [\alpha] +
\frac{1}{2} \text{Tr} \log \left[\frac{\alpha I}{\alpha I + A}\right] +
\alpha S - \frac{1}{2}\chi^2.
\end{equation}
```

The defining equation for the `classic MaxEnt` equation reads:

```math
-2\alpha S = \text{Tr} \left(\frac{\Lambda}{\alpha I + \Lambda}\right).
```

The summation on the right hand side of the above equation is defined to
be ``N_g``, the number of good measurements:

```math
\begin{equation}
N_g = \text{Tr} \left(\frac{\Lambda}{\alpha I + \Lambda}\right).
\end{equation}
```

If ``\lambda_i`` are the eigenvalues of ``\Lambda``, then

```math
\begin{equation}
N_g = \sum_i \frac{\lambda_i}{\alpha + \lambda_i}.
\end{equation}
```

**``\Lambda`` matrix**

For Shannon-Jaynes entropy,

```math
\begin{equation}
\frac{\partial^2 S[A]}{\partial A_i \partial A_j} =
-\delta_{ij}\frac{\Delta_i}{A_i} =
-\delta_{ij}\frac{\sqrt{\Delta_i \Delta_j}}{\sqrt{A_i A_j}},
\end{equation}
```

```math
\begin{equation}
\Lambda_{ij} = \sqrt{\frac{A_i}{\Delta_i}}
               \frac{\partial^2 L}{\partial A_i \partial A_j}
               \sqrt{\frac{A_j}{\Delta_j}}.
\end{equation}
```

```math
\begin{equation}
\frac{\partial^2 S[A^+,A^-]}{\partial A_i \partial A_j} =
-\delta_{ij}\frac{\Delta_i}{\sqrt{A_i^2 + 4m_i^2}} =
-\delta_{ij}\frac{\sqrt[4]{\Delta^2_i}\sqrt[4]{\Delta^2_j}}
                 {\sqrt[4]{A_i^2 + 4m_i^2} \sqrt[4]{A_j^2 + 4m_j^2}},
\end{equation}
```

```math
\begin{equation}
\Lambda_{ij} = \sqrt[4]{\frac{A^2_i + 4m_i^2}{\Delta^2_i}}
               \frac{\partial^2 L}{\partial A_i \partial A_j}
               \sqrt[4]{\frac{A^2_j + 4m_j^2}{\Delta^2_j}}.
\end{equation}
```

---

For Bayesian Reconstruction entropy,

```math
\begin{equation}
\frac{\partial^2 S[A]}{\partial A_i \partial A_j} =
-\delta_{ij} \frac{\Delta_i}{A^2_i} =
-\delta_{ij} \frac{\sqrt{\Delta_i \Delta_j}}{A_i A_j},
\end{equation}
```

```math
\begin{equation}
\Lambda_{ij} = \frac{A_i}{\sqrt{\Delta_i}}
               \frac{\partial^2 L}{\partial A_i \partial A_j}
               \frac{A_j}{\sqrt{\Delta_j}}.
\end{equation}
```

```math
\begin{equation}
\frac{\partial^2 S[A^+,A^-]}{\partial A_i \partial A_j} =
-\delta_{ij} X_{ij} Y_{ij},
\end{equation}
```

```math
\begin{equation}
X_{ij} = \frac{\sqrt{2\Delta_i} \sqrt{2\Delta_j}}{
        \left(\sqrt{A^2_i + m^2_i} + m_i + A_i\right)
        \left(\sqrt{A^2_j + m^2_j} + m_j + A_j\right)
    },
\end{equation}
```

```math
\begin{equation}
Y_{ij} =
    \frac{
        \sqrt{A_i + \sqrt{A^2_i + m^2_i}}
        \sqrt{A_j + \sqrt{A^2_j + m^2_j}}
    }{
        \sqrt[4]{A^2_i + m^2_i}
        \sqrt[4]{A^2_j + m^2_j}
    },
\end{equation}
```

```math
\begin{equation}
\Lambda_{ij} = Z_i \frac{\partial^2 L}{\partial A_i \partial A_j} Z_j
\end{equation}
```

```math
\begin{equation}
Z_i = \frac{\left(\sqrt{A^2_i + m^2_i} + m_i + A_i\right)}{\sqrt{2\Delta_i}}
      \frac{\sqrt[4]{A^2_i + m^2_i}}{\sqrt{A_i + \sqrt{A^2_i + m^2_i}}}
\end{equation}
```

```math
\begin{equation}
Z_j = \frac{\left(\sqrt{A^2_j + m^2_j} + m_j + A_j\right)}{\sqrt{2\Delta_j}}
      \frac{\sqrt[4]{A^2_j + m^2_j}}{\sqrt{A_j + \sqrt{A^2_j + m^2_j}}}
\end{equation}
```

**Reference:**

[1] G. J. Kraberger, *et al.*, Phys. Rev. B **96**, 155128 (2017).

[2] M. Jarrell, *et al.*, Phys. Rep. **269**, 133 (1996).
=#

"""
    calc_bayes(
        mec::MaxEntContext{R},
        A::Vector{R},
        S::R,
        χ²::R,
        α::R,
        stype::Stype
    ) where {R<:Real}

It calculates Bayesian convergence criterion (`ng`, `tr`, and `conv`) for
classic maxent (maximum of probablility distribution) and then Bayesian
a-posteriori probability (`log_prob`) for `α` after optimization of `A`.

Here, `A` is the spectral function, `S` the entropy, `χ²` the deviation,
and `α` weight factor of the entropy.

### Arguments
See above explanations.

### Returns
* ng -> -2.0αS.
* tr -> Tr(Λ / (αI + Λ)).
* conv -> Ratio between `ng` and `tr`.
* prob -> Pr[α | \\bar{G}].

See also: [`calc_bayes_od`](@ref).
"""
function calc_bayes(mec::MaxEntContext{R},
                    A::Vector{R},
                    S::R,
                    χ²::R,
                    α::R,
                    stype::Stype) where {R<:Real}
    if mec.stype isa SJ
        T = sqrt.(A ./ mec.δ)
    else
        T = A ./ sqrt.(mec.δ)
    end
    Λ = (T * T') .* mec.hess

    nsvd = size(mec.Vₛ, 2)
    λ = eigvals(Hermitian(Λ))[(end - nsvd + 1):end]
    filter!(x -> x > 0, λ)
    ng = -R(2) * α * S
    tr = sum(λ ./ (α .+ λ))
    conv = tr / ng

    eig_sum = sum(log.(α ./ (α .+ λ)))
    log_prob = α * S - R(1//2) * χ² + log(α) + R(1//2) * eig_sum

    return ng, tr, conv, exp(log_prob)
end

"""
    calc_bayes_od(
        mec::MaxEntContext{Q},
        A::Vector{Q},
        S::Q,
        χ²::Q,
        α::Q,
        stype::Stype
    ) where {Q<:Real}

It calculates Bayesian convergence criterion (`ng`, `tr`, and `conv`) for
classic maxent (maximum of probablility distribution) and then Bayesian
a-posteriori probability (`log_prob`) for `α` after optimization of `A`.

Here, `A` is the spectral function, `S` the entropy, `χ²` the deviation,
and `α` weight factor of the entropy.

It is just a offdiagonal version of `calc_bayes()`.

### Arguments
See above explanations.

### Returns
* ng -> -2.0αS.
* tr -> Tr(Λ / (αI + Λ)).
* conv -> Ratio between `ng` and `tr`.
* prob -> Pr[α | \\bar{G}].

See also: [`calc_bayes`](@ref).
"""
function calc_bayes_od(mec::MaxEntContext{Q},
                       A::Vector{Q},
                       S::Q,
                       χ²::Q,
                       α::Q,
                       stype::Stype) where {Q<:Real}
    if mec.stype isa SJ
        R = (A .^ 2 + 4 * mec.model .^ 2) ./ (mec.δ .^ 2)
        T = R .^ Q(0.25)
    else
        R = sqrt.(A .^ 2 + mec.model .^ 2)
        X = (R .+ mec.model .+ A) ./ sqrt.(2 * mec.δ)
        Y = sqrt.(R) ./ sqrt.(A .+ R)
        T = X .* Y
    end
    Λ = (T * T') .* mec.hess

    nsvd = size(mec.Vₛ, 2)
    λ = eigvals(Hermitian(Λ))[(end - nsvd + 1):end]
    filter!(x -> x > 0, λ)
    ng = -2 * α * S
    tr = sum(λ ./ (α .+ λ))
    conv = tr / ng

    eig_sum = sum(log.(α ./ (α .+ λ)))
    log_prob = α * S - Q(1//2) * χ² + log(α) + Q(1//2) * eig_sum

    return ng, tr, conv, exp(log_prob)
end

"""
    calc_chi2(mec::MaxEntContext{T}, A::Vector{T}) where {T<:Real}

It computes the χ²-deviation of the spectral function `A`.

### Arguments
* mec -> A MaxEntContext struct.
* A -> Spectral function.

### Returns
* χ² -> Goodness-of-fit functional.
"""
function calc_chi2(mec::MaxEntContext{T}, A::Vector{T}) where {T<:Real}
    Gₙ = reprod(mec.mesh, mec.kernel, A)
    χ² = sum(mec.σ⁻² .* ((mec.Gᵥ - Gₙ) .^ 2))
    return χ²
end
