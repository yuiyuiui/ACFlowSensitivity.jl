using ACFlow, DelimitedFiles, Plots
import ACFlowSensitivity: continous_spectral_density, generate_G_values_cont, generate_G_values_disc


μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];



A = continous_spectral_density(μ, σ, peak);

β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 801;
noise = 1e-2;
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
    "ngamm" => 1000 ,      # Number of δ functions. Their superposition is used to mimic the spectral functions.
    "nwarm" => 1000 ,      # nwarm = 1000
    "nstep" => 20000 ,     # Number of Monte Carlo sweeping steps.
    "ndump" => 200 ,       # Intervals for monitoring Monte Carlo sweeps. For every ndump steps, the StochSK solver will try to output some useful information to help diagnosis.
    "retry" => 10 ,        # How often to recalculate the goodness-of-fit function (it is actually χ² ) to avoid numerical deterioration.
    "theta" => 1e+6 ,      # Starting value for the θ parameter. The StochSK solver always starts with a huge θ parameter, and then decreases it gradually.
    "ratio" => 0.9 ,       # Scaling factor for the θ parameter. It should be less than 1.0.
);



setup_param(B, S);

mesh, reA, reG = solve(wn, GFV);

plot(mesh,A.(mesh),label = "origin",title = "noise = $noise")
plot!(mesh,reA,label = "re construct")




γ_vec=rand(N) * 5
GFV_disc = generate_G_values_disc(β,N,γ_vec;noise = noise)
mesh, reA, reG = solve(wn, GFV_disc)


plot(wn, γ_vec, seriestype=:stem, linecolor=:blue, marker=:circle, markersize=6, linestyle=:dash,label = "origin")
plot!(mesh,reA, label = "re construct")