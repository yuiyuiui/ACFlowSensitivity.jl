# ---------------------------------------------------
# smooth situation
# ---------------------------------------------------

using DelimitedFiles, Printf, ACFlow
using Plots,Random
using ACFlowSensitivity

β = 10.0;
N = 20;
Random.seed!(6)
μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
output_bound = 5.0;
output_number = 801;
output_range = range(-output_bound,output_bound,output_number);
output_range = collect(output_range);
wn = (collect(0:N-1) .+ 0.5) * 2π / β;


noise0 = 0.0
noise1 =1e-5
noise2 =1e-4
noise3 =1e-3
Gvalue0=generate_G_values_cont(β,N,A;noise=noise0);
Gvalue1=generate_G_values_cont(β,N,A;noise=noise1);
Gvalue2=generate_G_values_cont(β,N,A;noise=noise2);
Gvalue3=generate_G_values_cont(β,N,A;noise=noise3);


B = Dict{String,Any}(
    "solver" => "StochPX",  # Choose MaxEnt solver
    "mtype"  => "gauss",   # Default model function
    "mesh"   => "tangent", # Mesh for spectral function
    "ngrid"  => N,        # Number of grid points for input data
    "nmesh"  => output_number,       # Number of mesh points for output data
    "wmax"   => output_bound,       # Right boundary of mesh
    "wmin"   => -output_bound,      # Left boundary of mesh
    "beta"   => β,      # Inverse temperature
);


S = Dict{String,Any}(
    "method" => "mean",
    "nfine" => 100000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
    "npole" => 801,    # Number of poles on the real axis. These poles are used to mimic the Matsubara Green's function.
    "ntry" => 1000,   # Number of attempts to figure out the solution.
    "nstep" => 1000,    #  Number of Monte Carlo sweeping steps per attempt / try.
    "theta" => 1e+8,    # . Artificial inverse temperature θ. When it is increased, the transition probabilities of Monte Carlo updates will decrease.
    "eta" => 1e-3,
);
setup_param(B, S);

mesh, reA0_cont, _ = solve(wn, Gvalue0)
_   , reA1_cont, _ = solve(wn, Gvalue1)
_   , reA2_cont, _ = solve(wn, Gvalue2)
_   , reA3_cont, _ = solve(wn, Gvalue3)

plot(output_range, A.(output_range), label = "origin A(w)", title = "SPX for smooth type", xlabel = "w", ylabel = "A(w)")
plot!(output_range, reA1_cont, label = "reconstruct A1(w), noise: 1e-5", linewidth = 0.5)
plot!(output_range, reA2_cont, label = "reconstruct A2(w), noise: 1e-4", linewidth = 0.5)
plot!(output_range, reA3_cont, label = "reconstruct A3(w), noise: 1e-3", linewidth = 0.5)
plot!(output_range, reA0_cont, label = "reconstruct A0(w), noise: 0.0")








# ---------------------------------------------------
# descrete situation
# ---------------------------------------------------

using DelimitedFiles, Printf, ACFlow
using Plots,Random
import ACFlowSensitivity.generate_G_values_delta

β = 10.0;
N = 20;
Random.seed!(6)
output_bound = 5.0;
output_number = 200;
output_range = range(-output_bound,output_bound,output_number);
output_range = collect(output_range);
wn = (collect(0:N-1) .+ 0.5) * 2π / β;
noise0 = 0.0
noise1 =1e-5
noise2 =1e-4
noise3 =1e-3

poles = [-2.0,-1.0,1.0,2.0]
γ_vec = 0.25 * ones(length(poles))

Gval0 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise0)
Gval1 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise1)
Gval2 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise2)
Gval3 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise3)


B = Dict{String,Any}(
    "solver" => "StochPX",  # Choose MaxEnt solver
    "mtype"  => "gauss",   # Default model function
    "mesh"   => "tangent", # Mesh for spectral function
    "ngrid"  => N,        # Number of grid points for input data
    "nmesh"  => output_number,       # Number of mesh points for output data
    "wmax"   => output_bound,       # Right boundary of mesh
    "wmin"   => -output_bound,      # Left boundary of mesh
    "beta"   => β,      # Inverse temperature
);


S = Dict{String,Any}(
    "method" => "best",
    "nfine" => 100000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
    "npole" => 4,    # Number of poles on the real axis. These poles are used to mimic the Matsubara Green's function.
    "ntry" => 1000,   # Number of attempts to figure out the solution.
    "nstep" => 100000,    #  Number of Monte Carlo sweeping steps per attempt / try.
    "theta" => 1e+6,    # . Artificial inverse temperature θ. When it is increased, the transition probabilities of Monte Carlo updates will decrease.
    "eta" => 1e-2,
);
setup_param(B, S);

mesh, reA0_delta, _ = solve(wn, Gval0)
_   , reA1_delta, _ = solve(wn, Gval1)
_   , reA2_delta, _ = solve(wn, Gval2)
_   , reA3_delta, _ = solve(wn, Gval3)

plot(output_range, reA0_delta, label = "reconstruct A0(w), noise: 0.0", title = "SPX for delta type", xlabel = "w", ylabel = "A(w)")
plot!(output_range, reA1_delta, label = "reconstruct A1(w), noise: 1e-5", linewidth = 0.5)
plot!(output_range, reA2_delta, label = "reconstruct A2(w), noise: 1e-4", linewidth = 0.5)
plot!(output_range, reA3_delta, label = "reconstruct A3(w), noise: 1e-3", linewidth = 0.5)
plot!(poles, γ_vec, seriestype=:stem, linecolor=:blue, marker=:circle, markersize=3, linestyle=:dash,label = "origin poles")

