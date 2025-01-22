using ACFlow, DelimitedFiles, Plots
import ACFlowSensitivity: continous_spectral_density, generate_G_values_cont


μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];


#=
μ=6*rand(2).-3.0
σ=2*rand(2)
peak=rand(2)*4
=#

A = continous_spectral_density(μ, σ, peak);

β = 10.0;
N = 20;
output_bound = 8.0;
output_number = 801;
noise = 1e-2;
wn = collect((0:N-1) .+ 0.5) * 2π / β;
GFV = generate_G_values_cont(β, N, A; noise = noise);

B = Dict{String,Any}(
    "solver" => "MaxEnt",  # Choose MaxEnt solver
    "mtype"  => "gauss",   # Default model function
    "mesh"   => "tangent", # Mesh for spectral function
    "ngrid"  => N,        # Number of grid points for input data
    "nmesh"  => output_number,       # Number of mesh points for output data
    "wmax"   => output_bound,       # Right boundary of mesh
    "wmin"   => -output_bound,      # Left boundary of mesh
    "beta"   => β,      # Inverse temperature
);


S = Dict{String,Any}(
    "nalph"  => 15,        # Number of α parameters
    "alpha"  => 1e12,      # Starting value of α parameter
    "blur"   => -1.0,      # Enable preblur or not
);



setup_param(B, S);

mesh, reA, reG = solve(wn, GFV);

plot(mesh,A.(mesh))
plot!(mesh,reA)