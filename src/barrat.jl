# Aaa algorithm for continuous spectral density
struct BarRatFunc{T<:Number} <: Function
    w::Vector{T} # weights
    g::Vector{T} # grids
    v::Vector{T} # values
end

(f::BarRatFunc)(x) = sum((f.w .* f.v) ./ (x .- f.g))/sum(f.w ./ (x .- f.g))

function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::BarRat) where {T<:Real}
    wn = ctx.wn
    alg.denoisy &&
        (GFV = (alg.prony_tol>0 ? PronyApproximation(wn, GFV, alg.prony_tol)(wn) :
                PronyApproximation(wn, GFV)(wn)))
    brf, _ = aaa(ctx.iwn, GFV; alg=alg)
    ctx.spt isa Cont && return ctx.mesh, brf2A(brf, ctx)
    ctx.spt isa Delta &&
        return ctx.mesh, brf2A(brf, ctx), Poles(GFV, ctx.iwn, alg.pcut)(brf.w, brf.g)
    # For Mixed spectrum:
end

# aaa algorithm writen by myself
function aaa(grid::Vector{T}, values::Vector{T}; alg::BarRat) where {T}
    @assert length(grid)>0
    @assert length(grid)==length(values)

    # preparation
    tol = alg.aaa_tol
    minsgl = alg.minsgl
    max_degree = alg.max_degree
    lookaheaad = alg.lookaheaad
    m=length(grid)
    best_error=Inf
    best_n=0
    best_weight=T(0)
    best_index=Int[]

    max_values=maximum(abs.(values))
    wait_index=Set(1:m)
    chosen_index=Int64[]

    C=Matrix{T}(undef, m, m)
    L=Matrix{T}(undef, m, m)
    R=zeros(m)

    #get the first node
    mean_value=sum(values)/m
    _, idx = findmax(abs(value - mean_value) for value in values)

    push!(chosen_index, idx)
    delete!(wait_index, idx)

    # begin iteration
    n=0
    while true
        n+=1
        active_grid=grid[chosen_index]
        active_values=values[chosen_index]
        @inbounds @fastmath for i in wait_index
            δ = grid[i] - active_grid[n]
            C[i, n] = iszero(δ) ? 1 / eps(real(T)) : 1 / δ
            L[i, n] = (values[i] - active_values[n]) * C[i, n]
        end

        _, S, V=svd(L[collect(wait_index), 1:n])
        minsgl>S[1] && error("minsgl is too large, minsgl ∈ (S[end], S[1])")
        minsgl_idx = findfirst(reverse(S) .>= minsgl)
        w=V[:, end - minsgl_idx + 1]

        wait_active_C=view(C, collect(wait_index), 1:n)
        num=wait_active_C*(w .* active_values)
        den=wait_active_C*w
        R[collect(wait_index)]=abs.(values[collect(wait_index)]-num ./ den)
        error, next_index=findmax(R)

        # Do we choose this as best?
        if (error<best_error)
            best_error=error
            best_n=n
            best_weight=w
            best_index=copy(chosen_index)
        end

        # Do we end the iteration?
        if (best_error<tol*max_values) ||
           (n>=max_degree) || #n>=6 ||
           ((n-best_n>=lookaheaad)&&(best_error<1e-2*max_values))
            break
        end
        if n>=(m>>1)
            break
        end

        push!(chosen_index, next_index)
        delete!(wait_index, next_index)
        R[next_index]=0
    end
    return BarRatFunc(best_weight, grid[best_index], values[best_index]), best_index
end

function brf2A(brf::BarRatFunc{Complex{T}}, ctx::CtxData{T}) where {T}
    ctx.spt isa Cont && return -imag.(brf.(ctx.mesh))/T(π)
    ctx.spt isa Delta && return -imag.(brf.(ctx.mesh .+ im*ctx.η))/T(π)
    return error("Now only support continuous spectrum")
end

# deal with poles
"""
    bc_poles(f::BarRatFunc{T}) where {T}

Return the poles of the rational function `f`.

### Arguments
* f -> A BarRatFunc struct.

### Returns
* pole -> List of poles.
"""
function bc_poles(w::Vector{T}, g::Vector{T}) where {T}
    rT = real(T)
    nonzero = @. !iszero(w)
    g, w = g[nonzero], w[nonzero]
    m = length(w)
    B = diagm([zero(rT); ones(rT, m)])
    E = [zero(rT) transpose(w); ones(rT, m) diagm(g)]
    pole = [] # Put it into scope
    try
        pole = filter(isfinite, eigvals(E, B))
    catch
        # Generalized eigen not available in extended precision, so:
        λ = filter(g->abs(g)>1e-13, eigvals(E\B))
        pole = 1 ./ λ
    end
    return pole
end

struct Poles{T<:Complex,S<:Real}<:Function
    GFV::Vector{T}
    iwn::Vector{T}
    pcut::S
end
function (poles::Poles{T,S})(w::Vector{T}, g::Vector{T}) where {T,S}
    # Get positions of the poles
    p = bc_poles(w, g)
    # Print their positions
    println("Raw poles:")
    for i in eachindex(p)
        pᵢ = (real(p[i]), imag(p[i]))
        @show pᵢ
    end
    #
    # Filter unphysical poles
    filter!(z -> abs(imag(z)) < poles.pcut, p)
    if length(p) == 0
        error("The number of poles is zero. You should increase pcut")
    end
    #
    # Print their positions again
    println("New poles after filtering:")
    for i in eachindex(p)
        pᵢ = (real(p[i]), imag(p[i]))
        @show pᵢ
    end

    # Now we know positions of these poles, and we need to figure out
    # their amplitudes. This is a typical optimization problem. We just
    # employ the BFGS algorithm to do this job.
    γopt = poles2realγ(p, poles.GFV, poles.iwn)

    # Print their weights / amplitudes.
    println("New poles:")
    for i in eachindex(p)
        pᵢ = p[i]
        γᵢ = γopt[i]
        @show (pᵢ, γᵢ)
    end
    return (real(p), γopt)
end

#---------------------------------
# solve differentiation
function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::BarRat) where {T<:Real}
    alg.denoisy && error("denoisy is not supported for differentiation")
    alg.minsgl > 0 && error("minsgl is not supported for differentiation")
    brf, idx = aaa(ctx.iwn, GFV; alg=alg)
    reA = brf2A(brf, ctx)
    if ctx.spt isa Cont
        function fc(x)
            w, g, v = aaa4diff(x, idx, ctx.iwn)
            res = [-imag(sum((w .* v) ./ (ctx.mesh[i] .- g))/sum(w ./ (ctx.mesh[i] .- g)))/T(π)
                   for i in 1:length(ctx.mesh)]
            return res
        end
        reAdiff = Zygote.jacobian(fc, GFV)[1]
        return ctx.mesh, reA, reAdiff
    elseif ctx.spt isa Delta
        poles = Poles(GFV, ctx.iwn, alg.pcut)
        p, _ = poles(brf.w, brf.g)
        function fd(x)
            w, g, _ = aaa4diff(x, idx, ctx.iwn)
            nnz = @. !iszero(w)
            g1, w1 = g[nnz], w[nnz]
            return vcat(real(w1), imag(w1), real(g1), imag(g1))
        end
        ∂wgDiv∂G = Zygote.jacobian(fd, GFV)[1]
        nnz = @. !iszero(brf.w)
        pg, pw = brf.g[nnz], brf.w[nnz]
        m = length(pw)
        ∂pDiv∂wg = zeros(T, length(p), 4*m)
        for (i, pᵢ) in enumerate(p)
            tmp = sum(pw ./ (pg .- pᵢ) .^ 2)
            ∂pDiv∂wg[i, 1:m] = real(1 ./ (pᵢ .- pg) / tmp)
            ∂pDiv∂wg[i, (m + 1):(2 * m)] = real(im * ∂pDiv∂wg[i, 1:m])
            ∂pDiv∂wg[i, (2 * m + 1):(3 * m)] = real(pw ./ (pg .- pᵢ) .^ 2 / tmp)
            ∂pDiv∂wg[i, (3 * m + 1):(4 * m)] = real(im * ∂pDiv∂wg[i, (2 * m + 1):(3 * m)])
        end
        ∂pDiv∂G = ∂pDiv∂wg * ∂wgDiv∂G
        γ = pG2γ(p, GFV, poles.iwn)
        ∂γDiv∂p, ∂γDiv∂G = Zygote.jacobian((x, y) -> pG2γ(x, y, poles.iwn), p, GFV)
        return ctx.mesh, reA, (p, γ), (∂pDiv∂G, ∂γDiv∂p * ∂pDiv∂G + ∂γDiv∂G)
    else
        error("Now only support continuous and delta spectrum")
    end
end

function aaa4diff(GFV::Vector{Complex{T}}, idx::Vector{Int},
                  iwn::Vector{Complex{T}}) where {T<:Real}
    wait_idx = filter(i->i∉idx, 1:length(GFV))
    g = iwn[idx]
    v = GFV[idx]
    L = [((GFV[i] - GFV[j]) / (iwn[i] - iwn[j])) for i in wait_idx, j in idx]
    w = svd(L)[3][:, end]
    return w, g, v
end
