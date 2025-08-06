using ACFlowSensitivity, Plots, Random, LinearAlgebra
include("../test/testsetup.jl")#= BarRat for smooth type =#

function plot_alg_cont(alg::Solver; noise_num::Int=3, nwave::Int=2)
    T = Float64
    Random.seed!(6)
    noise_vec = [0.0, 1e-5, 1e-4, 1e-3, 1e-2][1:noise_num]
    μ = [T(1 // 2), T(-5 // 2), T(2.2)][1:nwave]
    σ = [T(1 // 5), T(4 // 5), T(1 // 3)][1:nwave]
    amplitudes = [T(1), T(3 // 10), T(2 // 5)][1:nwave]
    A, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh(), μ=μ, σ=σ, amplitudes=amplitudes)
    GFV_vec = Vector{Vector{Complex{T}}}(undef, length(noise_vec))
    GFV_vec[1] = GFV
    for i in 2:length(noise_vec)
        GFV_vec[i] = generate_GFV_cont(ctx.β, ctx.N, A; noise=noise_vec[i])
    end
    reA_vec = Vector{Vector{T}}(undef, length(noise_vec))
    for i in 1:length(noise_vec)
        _, reA_vec[i] = solve(GFV_vec[i], ctx, alg)
    end
    mesh = ctx.mesh.mesh

    p = plot(mesh,
             A.(mesh);
             label="origin A(w)",
             title="$(typeof(alg)) for cont type",
             xlabel="w",
             ylabel="A(w)",
             legend=:topleft)
    for i in 1:length(noise_vec)
        plot!(p,
              mesh,
              reA_vec[i];
              label="reconstruct A$i(w), noise: $(noise_vec[i])",
              linewidth=0.5)
    end
    return p
end

function plot_alg_delta(alg::Solver; noise_num::Int=1, fp_ww::Real=0.01, fp_mp::Real=0.1,
                        npole::Int=2)
    T = Float64
    Random.seed!(6)
    noise_vec = [0.0, 1e-5, 1e-4, 1e-3, 1e-2][1:noise_num]
    (orp, orγ), ctx, GFV = dfcfg(T, Delta(); mesh_type=TangentMesh(), fp_ww=fp_ww,
                                 fp_mp=fp_mp, npole=npole)
    GFV_vec = Vector{Vector{Complex{T}}}(undef, length(noise_vec))
    GFV_vec[1] = GFV
    for i in 2:length(noise_vec)
        GFV_vec[i] = generate_GFV_delta(ctx.β, ctx.N, orp, orγ; noise=noise_vec[i])
    end
    rep_vec = Vector{Vector{T}}(undef, length(noise_vec))
    reγ_vec = Vector{Vector{T}}(undef, length(noise_vec))
    for i in 1:length(noise_vec)
        _, _, (rep, reγ) = solve(GFV_vec[i], ctx, alg)
        rep_vec[i] = rep
        reγ_vec[i] = reγ
    end
    p = scatter(orp, fill(0.0, length(orp));
                title="$(typeof(alg)) for delta type",
                xlabel="w",
                ylabel="A(w)",
                label="original poles",
                markershape=:circle,
                markersize=4,
                markercolor=:red,
                xlims=(1, 3),
                ylims=(-0.02, 1.0))
    for i in 1:length(noise_vec)
        scatter!(rep_vec[i], fill(0.02, length(rep_vec[i]));
                 label="reconstruct poles, noise: $(noise_vec[i])",
                 markershape=:circle,
                 markersize=3,
                 markercolor=:blue)
    end
    return p
end

function ave_grad(J::Matrix{T}; try_num::Int=3000) where {T<:Number}
    res = zeros(real(T), size(J, 1))
    N = size(J, 2)
    for i in 1:try_num
        dx = randn(T, N)
        res += abs.(real(conj(J) * dx))
    end
    return res / try_num
end

function plot_errorbound_cont(alg::Solver; noise::Real=0.0, perm::Real=1e-4,
                              perm_num::Int=4, nwave::Int=2)
    T = Float64
    Random.seed!(6)
    μ = [T(1 // 2), T(-5 // 2), T(2.2)][1:nwave]
    σ = [T(1 // 5), T(4 // 5), T(1 // 3)][1:nwave]
    amplitudes = [T(1), T(3 // 10), T(2 // 5)][1:nwave]
    _, ctx, GFV = dfcfg(T, Cont(); mesh_type=TangentMesh(), noise=noise, μ=μ, σ=σ,
                        amplitudes=amplitudes)
    return plot_errorbound_cont(GFV, ctx, alg; perm=perm, perm_num=perm_num)
end

function plot_errorbound_delta(alg::Solver; noise::Real=0.0, perm::Real=1e-4,
                               perm_num::Int=4, npole::Int=2)
    T = Float64
    Random.seed!(6)
    (orp, orγ), ctx, GFV = dfcfg(T, Delta(); mesh_type=TangentMesh(), noise=noise,
                                 fp_ww=fp_ww,
                                 fp_mp=fp_mp, npole=npole)
    GFV_perm = Vector{Vector{ComplexF64}}(undef, perm_num)
    reA_perm = Vector{Vector{Float64}}(undef, perm_num)
    N = ctx.N
    for i in 1:perm_num
        GFV_perm[i] = GFV .+ randn(N) * perm .* exp.(1im * 2π * rand(N))
    end
    _, reA = solve(GFV, ctx, alg)
    for i in 1:perm_num
        _, reA_perm[i] = solve(GFV_perm[i], ctx, alg)
    end
    mesh = ctx.mesh.mesh
    p = plot(mesh,
             reA;
             label="reconstructed A(w)",
             title="error bound, $(typeof(alg)), Cont, perm: $(perm)",
             xlabel="w",
             ylabel="A(w)",
             legend=:topleft)
    for i in 1:perm_num
        plot!(p,
              mesh,
              reA_perm[i];
              label="permuted reA: $i",
              linewidth=0.5)
    end
    _, _, ∂reADiv∂G = solvediff(GFV, ctx, alg)
    ag = ave_grad(∂reADiv∂G) / 2
    Aupper = reA .+ perm * ag
    Alower = max.(0.0, reA .- perm * ag)
    plot!(p,
          mesh,
          Aupper;
          fillrange=Alower,
          fillalpha=0.3,
          label="Confidence region",
          linewidth=0)
    return p
end

function plot_errorbound_cont(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                              alg::Solver; perm::Real=1e-4, perm_num::Int=4) where {T<:Real}
    Random.seed!(6)
    GFV_perm = Vector{Vector{ComplexF64}}(undef, perm_num)
    reA_perm = Vector{Vector{Float64}}(undef, perm_num)
    N = ctx.N
    for i in 1:perm_num
        GFV_perm[i] = GFV .+ randn(N) * perm .* exp.(1im * 2π * rand(N))
    end
    _, reA = solve(GFV, ctx, alg)
    for i in 1:perm_num
        _, reA_perm[i] = solve(GFV_perm[i], ctx, alg)
    end
    mesh = ctx.mesh.mesh
    p = plot(mesh,
             reA;
             label="reconstructed A(w)",
             title="error bound, $(typeof(alg)), Cont, perm: $(perm)",
             xlabel="w",
             ylabel="A(w)",
             legend=:topleft)
    for i in 1:perm_num
        plot!(p,
              mesh,
              reA_perm[i];
              label="permuted reA: $i",
              linewidth=0.5)
    end
    _, _, ∂reADiv∂G = solvediff(GFV, ctx, alg)
    max_error = zeros(T, length(mesh))
    for i in 1:length(mesh)
        max_error[i] = perm * norm(∂reADiv∂G[i, :])
    end
    Aupper = reA .+ max_error
    Alower = max.(0.0, reA .- max_error)
    plot!(p,
          mesh,
          Aupper;
          fillrange=Alower,
          fillalpha=0.3,
          label="Confidence region",
          linewidth=0)
    return p
end
