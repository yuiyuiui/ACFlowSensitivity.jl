# ================================
# SpectrumType
# ================================

abstract type SpectrumType end

struct Cont <: SpectrumType end
struct Delta <: SpectrumType end
struct Mixed <: SpectrumType end

# ================================
# Context Data
# ================================

struct CtxData{T<:Real}
    spt::SpectrumType
    β::T
    N::Int
    wn::Vector{T}
    iwn::Vector{Complex{T}}
    mesh::Mesh{T}
    mesh_type::MeshMethod
    σ::T
    fp_ww::Real # find peaks window width
    fp_mp::Real # find peaks minimum peak height
    nproc::Int
    function CtxData(spt::SpectrumType,
                     β::T,
                     N::Int;
                     mesh_bound=ACFSDefults.mesh_bound[]::Real,
                     mesh_length=ACFSDefults.mesh_length[]::Int,
                     mesh_type::MeshMethod=ACFSDefults.mesh_type[]::MeshMethod,
                     σ::T=T(1e-4),
                     fp_ww::Real=T(0.01),
                     fp_mp::Real=T(0.1)) where {T<:Real}
        wn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β
        iwn = (collect(0:(N - 1)) .+ T(0.5)) * T(2π) / β * im
        mesh = make_mesh(T(mesh_bound), mesh_length, mesh_type)
        return new{T}(spt, β, N, wn, iwn, mesh, mesh_type, σ, fp_ww, fp_mp, nworkers())
    end
end

# ================================
# Solver
# ================================

abstract type Solver end

# BarRat ==========================
# abanda add singular values of the lowner matrix less than `minsgl` for numerical stability
# this method is under developing
struct BarRat <: Solver
    minsgl::Real
    aaa_tol::Real
    max_degree::Int
    lookaheaad::Int
    denoisy::Bool
    prony_tol::Real
    pcut::Real
    eta::Real
end

function BarRat(;
                minsgl::Real=0,
                aaa_tol::Real=ACFSDefults.tol[],
                max_degree::Int=150,
                lookaheaad::Int=10,
                denoisy::Bool=false,
                prony_tol::Real=-1,
                pcut::Real=1e-3,
                eta::Real=1e-4)
    return BarRat(minsgl, aaa_tol, max_degree, lookaheaad, denoisy, prony_tol, pcut, eta)
end

# NAC ==========================
mutable struct NAC <: Solver
    pick::Bool
    hardy::Bool
    hmax::Int
    alpha::Real
    eta::Real
end

function NAC(; pick=true,
             hardy=true,
             hmax=50,
             alpha=1e-4,
             eta=0.0,)
    @info("For delta type spectrum, `pick = false` and `hardy = false` are recommended")
    return NAC(pick, hardy, hmax, alpha, eta)
end

# MaxEnt ==========================
abstract type Stype end

struct SJ <: Stype end
struct BR <: Stype end

mutable struct MaxEnt <: Solver
    method::String
    stype::Stype
    nalph::Int
    alpha::Real
    ratio::Real
    model_type::String
    offdiag::Bool
    blur::Real
    test::Any
end
# what does `offdiag` in fact means?
function MaxEnt(;
                method::String="chi2kink",
                stype::Stype=SJ(),
                nalph::Int=16,
                alpha::Real=1e12,
                ratio::Real=10,
                model_type::String="Gaussian",
                offdiag::Bool=false,
                blur::Real=-1,
                test=missing)
    return MaxEnt(method, stype, nalph, alpha, ratio, model_type, offdiag, blur, test)
end

# SAN ==========================

mutable struct SAN <: Solver
    nfine::Int
    npole::Int
    nwarm::Int
    nstep::Int
    retry::Int
    θ::Real
    ratio::Real
    method::String
    nchain::Int
end
function SAN(npole::Int;
             nfine::Int=100000,
             nwarm::Int=1000,
             nstep::Int=20000,
             retry::Int=10,
             θ::Real=1e6,
             ratio::Real=0.9,
             method::String="chi2min",
             nchain::Int=nworkers())
    return SAN(nfine, npole, nwarm, nstep, retry, θ, ratio, method, nchain)
end

# SAC ==========================

struct SAC <: Solver
    nfine::Int
    npole::Int
    nwarm::Int
    nstep::Int
    ndump::Int
    nalph::Int
    alpha::Real
    ratio::Real
    nchain::Int
end
function SAC(npole::Int;
             nfine::Int=10000,
             nwarm::Int=4000,
             nstep::Int=4000000,
             ndump::Int=40000,
             nalph::Int=20,
             alpha::Real=1.0,
             ratio::Real=1.2,
             nchain::Int=nworkers())
    return SAC(nfine, npole, nwarm, nstep, ndump, nalph, alpha, ratio, nchain)
end

# SOM ==========================
struct SOM <: Solver
    ntry::Int
    nstep::Int
    nbox::Int
    sbox::Real
    wbox::Real
end
function SOM(;
             ntry::Int=3000,
             nstep::Int=10000,
             nbox::Int=100,
             sbox::Real=0.005,
             wbox::Real=0.02)
    return SOM(ntry, nstep, nbox, sbox, wbox)
end

# SPX ==========================
struct SPX <: Solver
    method::String
    nfine::Int
    npole::Int
    ntry::Int
    nstep::Int
    theta::Real
    eta::Real
end
function SPX(npole::Int;
             method::String="best",
             nfine::Int=100000,
             ntry::Int=1000,
             nstep::Int=10000,
             theta::Real=1e6,
             eta::Real=1e-4)
    return SPX(method, nfine, npole, ntry, nstep, theta, eta)
end

# ================================
# output format
function output_format(Aout::Vector{T}, GFV::Vector{Complex{T}}, ctx::CtxData{T},
                       alg::Solver) where {T<:Real}
    mesh = ctx.mesh.mesh
    if ctx.spt isa Delta
        p = mesh[find_peaks(mesh, Aout, ctx.fp_mp; wind=ctx.fp_ww)]
        hasfield(typeof(alg), :npole) && length(p) != alg.npole &&
            @warn("Number of poles is not correct")
        γ = pG2γ(p, GFV, ctx.iwn)
        return Aout, (p, γ)
    elseif ctx.spt isa Cont
        return Aout
    else
        error("Unsupported spectral function type")
    end
end

# multiple process
abstract type SST end # store information for sensitivity calculation
abstract type SMC end # Monte Carlo engine
function nproc_run!(alg::Solver, MC::SMC, args...)
    n = alg.nchain
    m = nworkers()
    STVEC = SST[]
    while n > 0
        println("Remaining chains to run: $n, current process number: $m")
        if nworkers() > 1
            nproc = min(n, m)
            println("Running in parallel mode, nworkers = $(nproc)")
            𝐹 = Future[]
            for i in 1:nproc
                MC.seed += RandomSeed1 + 1
                𝑓 = @spawnat i + 1 run!(MC, args..., alg)
                push!(𝐹, 𝑓)
            end
            for i in 1:nproc
                wait(𝐹[i])
                push!(STVEC, fetch(𝐹[i]))
            end
            n -= nproc
        else
            MC.seed += RandomSeed1 + 1
            ST = run!(deepcopy(MC), deepcopy.(args)..., alg)
            push!(STVEC, deepcopy(ST))
            n -= 1
        end
    end

    return STVEC
end

# solve differentiation
function pγdiff(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                alg::Solver; ns::Bool=false) where {T<:Real}
    @assert ctx.spt isa Delta
    N = ctx.N
    Aout, (rep, reγ) = solve(GFV, ctx, alg)
    n = length(rep)
    function f0(p, γ, G)
        @assert length(p) == length(γ)
        res = 0
        for j in 1:N
            res += abs2(sum(γ ./ (ctx.iwn[j] .- p)) - G[j])
        end
        return res
    end
    function f1(p, γ, G)
        @assert length(p) == length(γ) + 1
        res = 0
        for j in 1:N
            tmp = 0
            for k in 1:(n - 1)
                tmp += γ[k] / (ctx.iwn[j] - p[k])
            end
            tmp += (1 - sum(γ)) / (ctx.iwn[j] - p[n])
            res += abs2(tmp - G[j])
        end
        return res
    end
    if !ns # ns means Σ γ = 1
        f = f0
        repγ = vcat(rep, reγ)
    else
        f = f1
        repγ = vcat(rep, reγ[1:(end - 1)])
    end

    g(pγ, G) = f(pγ[1:n], pγ[(n + 1):end], G)
    g₁(pγ, G) = Zygote.gradient(pγ₁ -> g(pγ₁, G), pγ)[1]
    g₁₁(pγ, G) = Zygote.jacobian(pγ₁ -> g₁(pγ₁, G), pγ)[1]
    g₁₂(pγ, G) = Zygote.jacobian(G₁ -> g₁(pγ, G₁), G)[1]
    ∂pγDiv∂G = -pinv(g₁₁(repγ, GFV)) * g₁₂(repγ, GFV)
    Jp = ∂pγDiv∂G[1:n, :]
    Jγ = ∂pγDiv∂G[(n + 1):end, :]
    if ns
        Jγ = [Jγ; -sum(Jγ; dims=1)]
    end
    return Aout, (rep, reγ), (Jp, Jγ)
end

function Adiff(GFV::Vector{Complex{T}}, ctx::CtxData{T},
               alg::Solver; ns::Bool=false, diffonly::Bool=false) where {T<:Real}
    @assert ctx.spt isa Cont
    d = ctx.mesh.weight
    w = ctx.mesh.mesh
    wn = ctx.wn
    K = [d[k] / (im * wn[j] - w[k]) for j in 1:length(wn), k in 1:length(w)]
    ns && (K = [K; d'])
    Kʳ, Kⁱ = real(K), imag(K)
    K⁰ = (Kʳ' * Kʳ + Kⁱ' * Kⁱ)
    invK⁰ = pinv(K⁰)
    N = length(GFV)
    ∂ADiv∂G = invK⁰ * (Kʳ'[:, 1:N]) + invK⁰ * (Kⁱ'[:, 1:N]) * im
    diffonly && (return ∂ADiv∂G)
    Aout = solve(GFV, ctx, alg)
    return Aout, ∂ADiv∂G
end

# ================================
# Defaults
# ================================

module ACFSDefults
using ..ACFlowSensitivity
const tol = Ref(1e-12)
const mesh_bound = Ref(8)
const mesh_length = Ref(801)
const mesh_type = Ref(UniformMesh())
end
