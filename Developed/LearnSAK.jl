using ACFlow, DelimitedFiles
using Random, Plots
import ACFlowSensitivity: continous_spectral_density, generate_G_values_cont, generate_G_values_delta


μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];



A = continous_spectral_density(μ, σ, peak);

β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 801;
wn = collect((0:N-1) .+ 0.5) * 2π / β;
GFV = generate_G_values_cont(β, N, A; noise = noise);

B = Dict{String,Any}(
    "solver" => "StochSK",  # Choose MaxEnt solver
    "mesh"   => "tangent", # Mesh for spectral function
    "ngrid"  => N,        # Number of grid points for input data
    "nmesh"  => output_number,       # Number of mesh points for output data
    "wmax"   => output_bound,       # Right boundary of mesh
    "wmin"   => -output_bound,      # Left boundary of mesh
    "beta"   => β,      # Inverse temperature
);


S = Dict{String,Any}(
    "method" => "chi2min",
    "nfine" => 100000,     # Number of points of a very fine linear mesh. This mesh is for the δ functions.
    "ngamm" => 100 ,      # Number of δ functions. Their superposition is used to mimic the spectral functions.
    "nwarm" => 1000 ,      # nwarm = 1000
    "nstep" => 20000 ,     # Number of Monte Carlo sweeping steps.
    "ndump" => 200 ,       # Intervals for monitoring Monte Carlo sweeps. For every ndump steps, the StochSK solver will try to output some useful information to help diagnosis.
    "retry" => 10 ,        # How often to recalculate the goodness-of-fit function (it is actually χ² ) to avoid numerical deterioration.
    "theta" => 1e+6 ,      # Starting value for the θ parameter. The StochSK solver always starts with a huge θ parameter, and then decreases it gradually.
    "ratio" => 0.9 ,       # Scaling factor for the θ parameter. It should be less than 1.0.
);



setup_param(B, S);

mesh, reA, reG = solve(wn, GFV);

plot(mesh,A.(mesh),label = "origin",title = "Stochastic Poles Expansion, noise = 1e-3")
plot!(mesh,reA,label = "re construct")



poles_num = 4
poles = [-2.0,-1.0,1.0,2.0]
γ_vec=ones(poles_num) * 0.25
noise = 1e-3
GFV_disc = generate_G_values_delta(β,N,poles,γ_vec;noise = noise)
noise = 0.0
GFV_disc0 = generate_G_values_delta(β,N,poles,γ_vec;noise = noise)
mesh, reA, reG = solve(wn, GFV_disc)
mesh, reA0, reG0 = solve(wn, GFV_disc0)

plot(mesh,reA, title = "Stochastic Poles Expansion",label = "reconstruct poles, noise = 1e-3")
plot(mesh,reA0,label = "reconstruct poles, noise = 0.0")
plot!(poles, γ_vec, seriestype=:stem, linecolor=:blue, marker=:circle, markersize=6, linestyle=:dash,label = "origin poles")
