using ACFlowSensitivity, Plots, Random
include("../../test/testsetup.jl")
Random.seed!(6)

function plot_barrat_cont()
    T = Float64
    noise_vec = [0.0, 1e-5, 1e-4, 1e-3]
    A, ctx, GFV = dfcfg_cont(T; mesh_type=TangentMesh())
    GFV_vec = Vector{Vector{Complex{T}}}(undef, length(noise_vec))
    GFV_vec[1] = GFV
    for i in 2:length(noise_vec)
        GFV_vec[i] = generate_GFV_cont(ctx.β, ctx.N, A; noise=noise_vec[i])
    end
    reA_vec = Vector{Vector{T}}(undef, length(noise_vec))
    for i in 1:length(noise_vec)
        _, reA_vec[i] = solve(GFV_vec[i], ctx, BarRat(Cont()))
    end

    # draw the pictures
    p = plot(ctx.mesh,
             A.(ctx.mesh);
             label="origin A(w)",
             title="AAA for smooth type",
             xlabel="w",
             ylabel="A(w)")
    for i in 1:length(noise_vec)
        plot!(p,
              ctx.mesh,
              reA_vec[i];
              label="reconstruct A$i(w), noise: $(noise_vec[i])",
              linewidth=0.5)
    end
    return p
end

p = plot_barrat_cont()
