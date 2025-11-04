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
        reA_vec[i] = solve(GFV_vec[i], ctx, alg)
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
        _, (rep, reγ) = solve(GFV_vec[i], ctx, alg)
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

function plot_errorbound_cont(alg::Solver; noise::Real=0.0, perm::Real=1e-4,
                              perm_num::Int=4, nwave::Int=2, title::String="",
                              mesh_type::ACFlowSensitivity.MeshMethod=TangentMesh())
    T = Float64
    Random.seed!(6)
    μ = [T(1 // 2), T(-5 // 2), T(2.2)][1:nwave]
    σ = [T(1 // 5), T(4 // 5), T(1 // 3)][1:nwave]
    amplitudes = [T(1), T(3 // 10), T(2 // 5)][1:nwave]
    _, ctx, GFV = dfcfg(T, Cont(); mesh_type=mesh_type, noise=noise, μ=μ, σ=σ,
                        amplitudes=amplitudes)
    @show ctx
    return plot_errorbound_cont(GFV, ctx, alg; perm=perm, perm_num=perm_num, title=title)
end

function plot_errorbound_cont(GFV::Vector{Complex{T}}, ctx::CtxData{T},
                              alg::Solver; perm::Real=1e-4, perm_num::Int=4,
                              title::String="") where {T<:Real}
    Random.seed!(6)
    GFV_perm = Vector{Vector{ComplexF64}}(undef, perm_num)
    reA_perm = Vector{Vector{Float64}}(undef, perm_num)
    N = ctx.N
    for i in 1:perm_num
        noise = randn(N) .* exp.(1im * 2π * rand(N))
        noise .= noise / norm(noise) * perm
        GFV_perm[i] = GFV .+ noise
    end
    reA = solve(GFV, ctx, alg)
    for i in 1:perm_num
        reA_perm[i] = solve(GFV_perm[i], ctx, alg)
    end
    mesh = ctx.mesh.mesh
    title = title == "" ? "$(typeof(alg)), Cont-type, perm=$(perm)" : title
    p = plot(mesh,
             reA;
             label="Reconstructed A(w)",
             title=title,
             xlabel="w",
             ylabel="A(w)",
             legend=:topleft)
    for i in 1:perm_num
        plot!(p,
              mesh,
              reA_perm[i];
              label="Permuted reA: $i",
              linewidth=0.5)
    end
    _, ∂reADiv∂G = solvediff(GFV, ctx, alg)
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
          label="Variation region",
          linewidth=0)
    return p
end

function plot_errorbound_delta(alg::Solver; noise::Real=0.0, perm::Real=1e-4,
                               fp_ww::Real=0.01, fp_mp::Real=0.1,
                               perm_num::Int=4, p=[1.0, 2], γ=[0.5, 0.5],
                               title::String="",
                               mesh_type::ACFlowSensitivity.MeshMethod=TangentMesh())
    T = Float64
    Random.seed!(6)
    _, ctx, GFV = dfcfg(T, Delta(); mesh_type=mesh_type, noise=noise, poles=p, γ=γ,
                        fp_ww=fp_ww, fp_mp=fp_mp)
    GFV_perm = Vector{Vector{ComplexF64}}(undef, perm_num)
    reP = Vector{Tuple{T,T}}[]
    N = ctx.N
    for i in 1:perm_num
        noise = randn(N) .* exp.(1im * 2π * rand(N))
        noise .= noise / norm(noise) * perm
        GFV_perm[i] = GFV .+ noise
    end
    _, P0, (∂pDiv∂G, ∂γDiv∂G) = solvediff(GFV, ctx, alg)
    P = tv2vt(P0)
    @show P0
    @show P
    for i in 1:perm_num
        _, P10 = solve(GFV_perm[i], ctx, alg)
        P1 = tv2vt(P10)
        @show i
        @show P10
        @show P1
        push!(reP, P1)
    end
    D = Tuple{T,T}[]
    for i in 1:length(P)
        Di = (perm * norm(∂pDiv∂G[i, :]), perm * norm(∂γDiv∂G[i, :]))
        @show Di
        push!(D, Di)
    end

    return p = plot_points_with_regions(P, reP, D;
                                        title=title)
end

function tv2vt(v::Tuple{Vector{T},Vector{T}}) where {T<:Real}
    @assert length(v[1]) == length(v[2])
    res = Tuple{T,T}[]
    for i in 1:length(v[1])
        push!(res, (v[1][i], v[2][i]))
    end
    return res
end

function plot_points_with_regions(P::Vector{Tuple{T,T}},
                                  Pvec::Vector{Vector{Tuple{T,T}}},
                                  D::Vector{Tuple{T,T}};
                                  title::String="",
                                  colors::Vector{Symbol}=Symbol[]) where {T<:Real}

    # Check input parameters
    @assert length(P) == length(D) "The length of P and D must be equal"
    @assert length(P) > 0 "P cannot be empty"

    # Generate colors
    if isempty(colors)
        colors = [palette(:default, length(Pvec) + 1)...]
    end

    # Create plot
    p = plot(; title=title,
             xlabel="ω",
             ylabel="γ",
             legend=:topleft,
             grid=true,
             size=(800, 600))

    # Plot main points group P
    x_coords = [point[1] for point in P]
    y_coords = [point[2] for point in P]

    # Plot main points
    scatter!(p, x_coords, y_coords;
             label="Reconstructed poles and γ",
             color=:red,
             markersize=6,
             markerstrokewidth=2)

    # Plot main points region
    for i in 1:length(P)
        p_i, r_i = P[i]
        DX_i, DY_i = D[i]

        # Calculate the four corners of the square region
        x_left = p_i - DX_i
        x_right = p_i + DX_i
        y_bottom = r_i - DY_i
        y_top = r_i + DY_i

        # Plot square region (rectangle)
        plot!(p, [x_left, x_right, x_right, x_left, x_left],
              [y_bottom, y_bottom, y_top, y_top, y_bottom];
              color=:red,
              linewidth=2,
              alpha=0.3,
              fill=true,
              fillalpha=0.1,
              label=i == 1 ? "Variation Region" : "")

        # Add coordinate axis dashed lines
        # Vertical line (x-axis direction)
        plot!(p, [p_i, p_i], [0, r_i];
              color=:gray,
              linestyle=:dash,
              linewidth=1, label="")

        # Horizontal line (y-axis direction)
        plot!(p, [0, p_i], [r_i, r_i];
              color=:gray,
              linestyle=:dash,
              linewidth=1, label="")

        # Add coordinate annotations
        annotate!(p, p_i, -0.1, text("$(round(p_i, digits=2))", 8, :center))
        annotate!(p, -0.1, r_i, text("$(round(r_i, digits=2))", 8, :center))

        # Add DX and DY annotations
        # DX annotation (at the bottom of the square region)
        annotate!(p, p_i, y_bottom - 0.1, text("DX=$(round(DX_i, digits=2))", 8, :center))
        # DY annotation (at the left of the square region)
        annotate!(p, x_left - 0.1, r_i, text("DY=$(round(DY_i, digits=2))", 8, :center))
    end

    # Plot other points groups Pvec
    for (group_idx, point_group) in enumerate(Pvec)
        if !isempty(point_group)
            x_coords_group = [point[1] for point in point_group]
            y_coords_group = [point[2] for point in point_group]

            # Plot each points group with different colors
            color_idx = (group_idx % length(colors)) + 1
            current_color = colors[color_idx]

            scatter!(p, x_coords_group, y_coords_group;
                     label="Perturbed poles and γ $(group_idx)",
                     color=current_color,
                     markersize=4,
                     markerstrokewidth=1)
        end
    end

    return p
end
