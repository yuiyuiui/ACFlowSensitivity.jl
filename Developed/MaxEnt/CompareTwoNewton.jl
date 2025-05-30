using ACFlowSensitivity, Plots
using ACFlow, DelimitedFiles

μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 8.0;
output_number = 801;
noise = 0.0;
Gvalue = generate_G_values_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
wn = (collect(0:(N-1)) .+ 0.5) * 2π / β;
iwn = wn * im;

Aout = my_chi2kink(iwn, Gvalue, output_range)
plot(output_range, A.(output_range), label = "origin Spectral ", title = "noise=$noise")
plot!(output_range, Aout, label = "reconstruct Spectral")

B = Dict{String,Any}(
    "solver" => "MaxEnt",  # Choose MaxEnt solver
    "mtype" => "gauss",   # Default model function
    "mesh" => "tangent", # Mesh for spectral function
    "ngrid" => 20,        # Number of grid points for input data
    "nmesh" => 801,       # Number of mesh points for output data
    "wmax" => 8.0,       # Right boundary of mesh
    "wmin" => -8.0,      # Left boundary of mesh
    "beta" => 10.0,      # Inverse temperature
);


S = Dict{String,Any}(
    "nalph" => 15,        # Number of α parameters
    "alpha" => 1e12,      # Starting value of α parameter
    "blur" => -1.0,      # Enable preblur or not
);

setup_param(B, S);

mesh, reA, reG = solve(wn, Gvalue);
plot!(mesh, A.(mesh), label = "origin Spectral ", title = "noise=$noise")
plot(mesh, reA, label = "reconstruct Spectral by ACFlow")
plot!(output_range, Aout, label = "reconstruct Spectral by myself")
