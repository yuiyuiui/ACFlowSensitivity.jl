using ACFlow, DelimitedFiles, Plots
import ACFlowSensitivity: continous_spectral_density, generate_G_values_cont


μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];

A = continous_spectral_density(μ, σ, peak);

β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 801;
noise = 0.0;
wn = collect((0:N-1) .+ 0.5) * 2π / β;
GFV = generate_G_values_cont(β, N, A; noise = noise);

B = Dict{String,Any}(
    "solver" => "MaxEnt",  # Choose MaxEnt solver
    "mtype"  => "gauss",   # Default model function
    "mesh"   => "tangent", # Mesh for spectral function
    "ngrid"  => 20,        # Number of grid points for input data
    "nmesh"  => 801,       # Number of mesh points for output data
    "wmax"   => 8.0,       # Right boundary of mesh
    "wmin"   => -8.0,      # Left boundary of mesh
    "beta"   => 10.0,      # Inverse temperature
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