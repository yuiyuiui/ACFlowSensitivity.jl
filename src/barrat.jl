# Aaa algorithm for continuous spectral density
struct BarRatFunc{T<:Number} <: Function
    w::Vector{T}
    g::Vector{T}
    v::Vector{T}
end

(f::BarRatFunc)(x) = sum((f.w .* f.v) ./ (x .- f.g))/sum(f.w ./ (x .- f.g))

function solve(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::BarRat) where {T<:Real}
    wn = ctx.wn
    alg.denoisy &&
        (GFV = (alg.prony_tol>0 ? PronyApproximation(wn, GFV, alg.prony_tol)(wn) :
                PronyApproximation(wn, GFV)(wn)))
    brf, _ = aaa(ctx.iwn, GFV; alg=alg)
    reA = extract_spectrum(brf, ctx.mesh, alg, alg.spt)
    return ctx.mesh, reA
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

function extract_spectrum(brf::BarRatFunc, mesh::Vector{T}, alg::BarRat,
                          spt::Cont) where {T}
    return -imag.(brf.(mesh))/T(π)
end

#---------------------------------
# solve differentiation

function solvediff(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::BarRat) where {T<:Real}
    alg.denoisy && error("denoisy is not supported for differentiation")
    alg.minsgl > 0 && error("minsgl is not supported for differentiation")
    _, idx = aaa(ctx.iwn, GFV; alg=alg)
    reA = aaa4diff(GFV, idx, ctx)
    reAdiff = Zygote.jacobian(x -> aaa4diff(x, idx, ctx), GFV)[1]
    return ctx.mesh, reA, reAdiff, ∇L2loss(reAdiff, ctx.mesh_weights)[2]
end

function aaa4diff(GFV::Vector{Complex{T}}, idx::Vector{Int},
                  ctx::CtxData{T}) where {T<:Real}
    wait_idx = filter(i->i∉idx, 1:ctx.N)
    iwn = ctx.iwn
    mesh = ctx.mesh
    g = iwn[idx]
    v = GFV[idx]
    L = [((GFV[i] - GFV[j]) / (iwn[i] - iwn[j])) for i in wait_idx, j in idx]
    w = svd(L)[3][:, end]
    res = [-imag(sum((w .* v) ./ (mesh[i] .- g))/sum(w ./ (mesh[i] .- g)))/T(π)
           for i in 1:length(mesh)]
    return res
end
