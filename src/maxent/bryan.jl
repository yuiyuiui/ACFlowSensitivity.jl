
"""
    bryan(mec::MaxEntContext)

Apply the bryan algorithm to solve the analytic continuation problem.

Bryan's maxent calculates an average of spectral functions, weighted by
their Bayesian probability.

For the bryan algorithm, `alpha` is usually 500, and `ratio` is 1.1.
It is incompatible with the Bayesian Reconstruction entropy.

### Arguments
* mec -> A MaxEntContext struct.

### Returns
* svec -> A vector of dictionaries. It contains the intermediate solutions.
* sol -> Dictionary. It contains the final solution.

See also: [`MaxEntContext`](@ref).
"""
function bryan(mec::MaxEntContext)
    println("Apply bryan algorithm to determine optimized α")

    use_bayes = true
    alpha = get_m("alpha")
    ratio = get_m("ratio")
    n_svd = length(mec.Bₘ)
    nmesh = length(mec.mesh)

    u_vec = zeros(F64, n_svd)
    s_vec = []

    maxprob = 0.0
    while true
        sol = optimizer(mec, alpha, u_vec, use_bayes)
        push!(s_vec, sol)
        alpha = alpha / ratio
        @. u_vec = sol[:u]
        prob = sol[:prob]
        if prob > maxprob
            maxprob = prob
        elseif prob < 0.01 * maxprob
            break
        end
    end

    α_vec = map(x->x[:α], s_vec)
    p_vec = map(x->x[:prob], s_vec)
    p_vec = -p_vec ./ trapz(α_vec, p_vec)
    A_vec = map(x->x[:A], s_vec)

    nprob = length(p_vec)
    A_opt = zeros(F64, nmesh)
    spectra = zeros(F64, nmesh, nprob)
    for i in 1:nprob
        spectra[:, i] = A_vec[i] * p_vec[i]
    end
    for j in 1:nmesh
        A_opt[j] = -trapz(α_vec, spectra[j, :])
    end

    return A_opt
end
