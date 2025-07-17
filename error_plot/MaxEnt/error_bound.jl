using ACFlowSensitivity, Plots, Random, LinearAlgebra
Random.seed!(6)

function ave_grad(J::Matrix{T}; try_num::Int=1000) where {T<:Number}
    res = zeros(real(T), size(J, 1))
    N = size(J, 2)
    for i in 1:try_num
        dx = randn(T, N)
        res += abs.(real(conj(J)*dx))
    end
    return res / try_num
end

function error_bound_cont(;
                          μ=[0.5, -2.5],
                          σ=[0.2, 0.8],
                          amplitudes=[1.0, 0.3],
                          mesh_type=TangentMesh(),
                          β=10.0,
                          N=20,
                          noise=2e-5,
                          mb=8,
                          ml=801,
                          perm=1e-4,
                          permu_num=4)
    ctx = CtxData(Cont(), β, N; mesh_bound=mb, mesh_length=ml, mesh_type=mesh_type)
    A=continous_spectral_density(μ, σ, amplitudes)
    GFV = generate_GFV_cont(β, N, A; noise=noise)
    GFV_perm = Vector{Vector{ComplexF64}}(undef, permu_num)
    reA_perm = Vector{Vector{Float64}}(undef, permu_num)

    for i in 1:permu_num
        GFV_perm[i] = GFV .+ randn(N) * perm .* exp.(1im * 2π * rand(N))
    end
    _, reA = solve(GFV, ctx, MaxEntChi2kink())
    for i in 1:permu_num
        _, reA_perm[i] = solve(GFV_perm[i], ctx, MaxEntChi2kink())
    end
    p = plot(ctx.mesh,
             reA;
             label="reconstructed A(w)",
             title="error bound, MaxEnt, Cont, perm: $(perm)",
             xlabel="w",
             ylabel="A(w)")
    for i in 1:permu_num
        plot!(p,
              ctx.mesh,
              reA_perm[i];
              label="permuted reA: $i",
              linewidth=0.5)
    end
    _, _, ∂reADiv∂G, _ = solvediff(GFV, ctx, MaxEntChi2kink(; model_type="Gaussian"))
    ag = ave_grad(∂reADiv∂G)
    Aupper = reA .+ perm * ag
    Alower = max.(0.0, reA .- perm * ag)
    plot!(p,
          ctx.mesh,
          Aupper;
          fillrange=Alower,
          fillalpha=0.3,
          label="Confidence region",
          linewidth=0)
    return p
end

error_bound_cont()
