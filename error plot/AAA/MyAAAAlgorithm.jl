using ACFlowSensitivity, Plots, Random


#μ=3*rand(2).-1.5;σ=rand(2);peak=4*rand(2);
μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 801;
noise0 = 0.0;
noise1 = 1e-5;
noise2 = 1e-4;
noise3 = 1e-3;
Random.seed!(6)

Amesh, reconstruct_A0, _ = aaa_check(A; β, N, output_bound, output_number, noise = noise0);
_, reconstruct_A1, _ = aaa_check(A; β, N, output_bound, output_number, noise = noise1);
_, reconstruct_A2, _ = aaa_check(A; β, N, output_bound, output_number, noise = noise2);
_, reconstruct_A3, _ = aaa_check(A; β, N, output_bound, output_number, noise = noise3);

# draw the pictures
plot(
    Amesh,
    A.(Amesh),
    label = "origin A(w)",
    title = "AAA for smooth type",
    xlabel = "w",
    ylabel = "A(w)",
)
plot!(
    Amesh,
    reconstruct_A1.(Amesh),
    label = "reconstruct A1(w), noise: 1e-5",
    linewidth = 0.5,
)
plot!(
    Amesh,
    reconstruct_A2.(Amesh),
    label = "reconstruct A2(w), noise: 1e-4",
    linewidth = 0.5,
)
plot!(
    Amesh,
    reconstruct_A3.(Amesh),
    label = "reconstruct A3(w), noise: 1e-3",
    linewidth = 0.5,
)
plot!(
    Amesh,
    reconstruct_A0.(Amesh),
    label = "reconstruct A0(w), noise: 0.0",
    linewidth = 0.6,
)



#-------------------------


#Examples of find poles by ACFlow barycenteric

using DelimitedFiles, Printf, ACFlow
using Plots, Random
import ACFlowSensitivity.generate_G_values_delta


#descrete situation
β = 10.0;
N = 20;
Random.seed!(6)
output_number = 801;
output_bound = 5.0;
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
wn = (collect(0:(N-1)) .+ 0.5) * 2π / β;
noise0 = 0.0
noise1 = 1e-5
noise2 = 1e-4
noise3 = 1e-3
poles = [-2.0, -1.0, 1.0, 2.0]
γ_vec = 0.25 * ones(length(poles))

Gval0 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise0)
Gval1 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise1)
Gval2 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise2)
Gval3 = generate_G_values_delta(β, N, poles, γ_vec; noise = noise3)


B = Dict{String,Any}(
    "solver" => "BarRat",  # Choose MaxEnt solver
    "mtype" => "gauss",   # Default model function
    "mesh" => "tangent", # Mesh for spectral function
    "ngrid" => 20,        # Number of grid points for input data
    "nmesh" => 801,       # Number of mesh points for output data
    "wmax" => 8.0,       # Right boundary of mesh
    "wmin" => -8.0,      # Left boundary of mesh
    "beta" => β,      # Inverse temperature
);

S = Dict{String,Any}(
    "atype"=>"delta",
    "denoise"=>"none",
    "epsilon"=>1e-10,
    "pcut"=>0.99,
    "eta"=>1e-2,
);
setup_param(B, S);

mesh, reA0, _ = solve(wn, Gval0)
_, reA1, _ = solve(wn, Gval1)
_, reA2, _ = solve(wn, Gval2)
_, reA3, _ = solve(wn, Gval3)

plot(
    mesh,
    reA0,
    label = "reconstruct A0(w), noise: 0.0",
    title = "AAA for delta type",
    xlabel = "w",
    ylabel = "A(w)",
)
plot!(mesh, reA1, label = "reconstruct A1(w), noise: 1e-5", linewidth = 0.5)
plot!(mesh, reA2, label = "reconstruct A2(w), noise: 1e-4", linewidth = 0.5)
plot!(mesh, reA3, label = "reconstruct A3(w), noise: 1e-3", linewidth = 0.5)
plot!(
    poles,
    γ_vec,
    seriestype = :stem,
    linecolor = :blue,
    marker = :circle,
    markersize = 3,
    linestyle = :dash,
    label = "origin poles",
)
