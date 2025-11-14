using ACFlowSensitivity, CairoMakie, Random, LinearAlgebra
include("../test/testsetup.jl")

const title_size = 32
const label_size = 24

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

    fig = Figure()
    ax = Axis(fig[1, 1];
              title="$(typeof(alg)) for cont type",
              xlabel="ω",
              ylabel="A(ω)",
              titlesize=title_size,
              xlabelsize=label_size,
              ylabelsize=label_size)

    # Plot original data
    lines!(ax, mesh, A.(mesh); label="origin A(ω)", linewidth=1.5, color=:black)

    # Plot reconstructed data
    colors = [:blue, :red, :green, :orange, :purple]
    for i in 1:length(noise_vec)
        lines!(ax, mesh, reA_vec[i];
               label="reconstruct A$i(ω), noise: $(noise_vec[i])",
               linewidth=1.0,
               color=colors[mod1(i, length(colors))])
    end

    axislegend(ax; position=:lt)
    return fig
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
    fig = Figure()
    ax = Axis(fig[1, 1];
              title="$(typeof(alg)) for delta type",
              xlabel="ω",
              ylabel="A(ω)",
              limits=((1, 3), (-0.02, 1.0)),
              titlesize=title_size,
              xlabelsize=label_size,
              ylabelsize=label_size)

    # Plot original poles
    scatter!(ax, orp, fill(0.0, length(orp));
             label="original poles",
             markersize=12,
             color=:red,
             marker=:circle)

    # Plot reconstructed poles
    colors = [:blue, :green, :orange, :purple, :brown]
    for i in 1:length(noise_vec)
        scatter!(ax, rep_vec[i], fill(0.02, length(rep_vec[i]));
                 label="reconstruct poles, noise: $(noise_vec[i])",
                 markersize=10,
                 color=colors[mod1(i, length(colors))],
                 marker=:circle)
    end

    axislegend(ax)
    return fig
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
    title_str = title == "" ? "$(typeof(alg)), Cont-type, perm=$(perm)" : title

    fig = Figure()
    ax = Axis(fig[1, 1];
              title=title_str,
              xlabel="ω",
              ylabel="A(ω)",
              xgridvisible=false,
              ygridvisible=false,
              titlesize=title_size,
              xlabelsize=label_size,
              ylabelsize=label_size)

    # Plot reconstructed A(ω)
    lines!(ax, mesh, reA; label="Reconstructed A(ω)", linewidth=1.2, color=:black)

    # Plot permuted reconstructions
    colors = [:blue, :yellow, :green, :orange, :purple]
    for i in 1:perm_num
        lines!(ax, mesh, reA_perm[i];
               label="Permuted reA: $i",
               linewidth=0.7,
               color=colors[mod1(i, length(colors))])
    end

    # Calculate and plot error bounds
    _, ∂reADiv∂G = solvediff(GFV, ctx, alg)
    max_error = zeros(T, length(mesh))
    for i in 1:length(mesh)
        max_error[i] = perm * norm(∂reADiv∂G[i, :])
    end
    Aupper = reA .+ max_error
    Alower = max.(0.0, reA .- max_error)

    # Plot variation region using band
    band!(ax, mesh, Alower, Aupper;
          label="Variation region",
          color=(:steelblue, 0.4))

    # Set y-axis to start from 0
    ymin = min(0.0, minimum(Alower))
    ymax = maximum(Aupper)*1.1
    ylims!(ax, ymin, ymax)

    axislegend(ax; position=:lt)
    return fig
end

function plot_errorbound_delta(alg::Solver; noise::Real=0.0, perm::Real=1e-4,
                               fp_ww::Real=0.01, fp_mp::Real=0.1,
                               perm_num::Int=4, p=[1.0, 2], γ=[0.5, 0.5],
                               title::String="",
                               mesh_type::ACFlowSensitivity.MeshMethod=TangentMesh(),
                               mb::Real=8.0, ml::Int=801)
    T = Float64
    Random.seed!(6)
    _, ctx, GFV = dfcfg(T, Delta(); mesh_type=mesh_type, noise=noise, poles=p, γ=γ,
                        fp_ww=fp_ww, fp_mp=fp_mp, mb=mb, ml=ml)
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
        default_colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]
        colors = default_colors[1:min(length(Pvec) + 1, length(default_colors))]
    end

    # Create plot
    fig = Figure(; size=(800, 600))

    # Set fixed display limits
    xlims = (0.5, 2.5)
    ylims = (0.2, 0.7)

    @show label_size
    @show title_size

    ax = Axis(fig[1, 1];
              title=title,
              xlabel="ω",
              ylabel="γ",
              limits=(xlims, ylims),
              xgridvisible=false,
              ygridvisible=false,
              titlesize=title_size,
              xlabelsize=label_size,
              ylabelsize=label_size)

    # Plot main points group P
    x_coords = [point[1] for point in P]
    y_coords = [point[2] for point in P]

    # Filter out NaN and Inf values
    valid_indices = [i
                     for i in 1:length(x_coords)
                     if isfinite(x_coords[i]) && isfinite(y_coords[i])]
    x_coords_valid = [x_coords[i] for i in valid_indices]
    y_coords_valid = [y_coords[i] for i in valid_indices]

    # Plot main points
    if !isempty(x_coords_valid)
        scatter!(ax, x_coords_valid, y_coords_valid;
                 label="Reconstructed poles and γ",
                 color=:red,
                 markersize=12,
                 marker=:circle,
                 strokewidth=2)
    end

    # Plot main points region
    for i in 1:length(P)
        p_i, r_i = P[i]
        DX_i, DY_i = D[i]

        # Skip if any value is NaN or Inf
        if !(isfinite(p_i) && isfinite(r_i) && isfinite(DX_i) && isfinite(DY_i))
            continue
        end

        # Calculate the four corners of the square region
        x_left = p_i - DX_i
        x_right = p_i + DX_i
        y_bottom = r_i - DY_i
        y_top = r_i + DY_i

        # Check if all corners are finite
        if !(isfinite(x_left) && isfinite(x_right) && isfinite(y_bottom) && isfinite(y_top))
            continue
        end

        # Plot square region (rectangle) using poly
        rect_points = Point2f[(x_left, y_bottom), (x_right, y_bottom),
                              (x_right, y_top), (x_left, y_top)]
        poly!(ax, rect_points;
              color=(:red, 0.1),
              strokecolor=:red,
              strokewidth=2,
              label=i == 1 ? "Variation Region" : "")

        # Add coordinate axis dashed lines
        # Vertical line (x-axis direction)
        if isfinite(p_i) && isfinite(r_i) && r_i != 0
            lines!(ax, [p_i, p_i], [0, r_i];
                   color=:gray,
                   linestyle=:dash,
                   linewidth=1)
        end

        # Horizontal line (y-axis direction)
        if isfinite(p_i) && isfinite(r_i) && p_i != 0
            lines!(ax, [0, p_i], [r_i, r_i];
                   color=:gray,
                   linestyle=:dash,
                   linewidth=1)
        end

        # Add coordinate annotations
        if isfinite(p_i)
            text!(ax, "$(round(p_i, digits=2))"; position=(p_i, -0.1),
                  align=(:center, :top), fontsize=8)
        end
        if isfinite(r_i)
            text!(ax, "$(round(r_i, digits=2))"; position=(-0.1, r_i),
                  align=(:right, :center), fontsize=8)
        end

        # Add DX and DY annotations
        # DX annotation (at the bottom of the square region)
        if isfinite(p_i) && isfinite(y_bottom) && isfinite(DX_i)
            text!(ax, "DX=$(round(DX_i, digits=2))"; position=(p_i, y_bottom - 0.1),
                  align=(:center, :top), fontsize=8)
        end
        # DY annotation (at the left of the square region)
        if isfinite(x_left) && isfinite(r_i) && isfinite(DY_i)
            text!(ax, "DY=$(round(DY_i, digits=2))"; position=(x_left - 0.1, r_i),
                  align=(:right, :center), fontsize=8)
        end
    end

    # Plot other points groups Pvec
    for (group_idx, point_group) in enumerate(Pvec)
        if !isempty(point_group)
            x_coords_group = [point[1] for point in point_group]
            y_coords_group = [point[2] for point in point_group]

            # Filter out NaN and Inf values
            valid_indices = [i
                             for i in 1:length(x_coords_group)
                             if isfinite(x_coords_group[i]) && isfinite(y_coords_group[i])]
            x_coords_valid = [x_coords_group[i] for i in valid_indices]
            y_coords_valid = [y_coords_group[i] for i in valid_indices]

            if !isempty(x_coords_valid)
                # Plot each points group with different colors
                color_idx = mod1(group_idx, length(colors))
                current_color = colors[color_idx]

                scatter!(ax, x_coords_valid, y_coords_valid;
                         label="Perturbed poles and γ $(group_idx)",
                         color=current_color,
                         markersize=8,
                         marker=:circle,
                         strokewidth=1)
            end
        end
    end

    # Only create legend if there are valid plots
    try
        axislegend(ax; position=:lt)
    catch e
        @warn "Failed to create legend: $e"
    end
    return fig
end
