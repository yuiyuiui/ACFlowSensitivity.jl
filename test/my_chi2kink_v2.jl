using ACFlowSensitivity
using ACFlow, DelimitedFiles, Plots
using Plots, LinearAlgebra, Random


Random.seed!(3)
β=10.0;
output_bound=5.0;
output_number=801;
noise = 1e-4;
output_range=range(-output_bound, output_bound, output_number);
output_range=collect(output_range);

############### test for continous spectral density
μ=[0.5, -2.5];
σ=[0.2, 0.8];
peak=[1.0, 0.3];
A=continous_spectral_density(μ, σ, peak);
N=20;
iwn=(collect(0:(N-1)) .+ 0.5)*2π/β * im;
wn = collect((0:(N-1)) .+ 0.5) * 2π / β;
Gvalue=generate_GFV_cont(β, N, A; noise = noise);
Aout=my_chi2kink(iwn, Gvalue, output_range)
plot(output_range, A.(output_range), label = "origin Spectral ", title = "noise=$noise")
plot!(output_range, Aout, label = "reconstruct Spectral")

B = Dict{String,Any}(
    "solver" => "MaxEnt",  # Choose MaxEnt solver
    "mtype" => "gauss",   # Default model function
    "mesh" => "tangent", # Mesh for spectral function
    "ngrid" => N,        # Number of grid points for input data
    "nmesh" => output_number,       # Number of mesh points for output data
    "wmax" => output_bound,       # Right boundary of mesh
    "wmin" => -output_bound,      # Left boundary of mesh
    "beta" => β,      # Inverse temperature
);
S = Dict{String,Any}(
    "nalph" => 15,        # Number of α parameters
    "alpha" => 1e12,      # Starting value of α parameter
    "blur" => -1.0,      # Enable preblur or not
);
setup_param(B, S);
mesh, reA, reG = solve(wn, Gvalue);
plot!(mesh, reA, label = "ACFlow MaxEnt")


model_ite = 20
Aout_v2 = ACFlowSensitivity.my_chi2kink_v2(iwn, Gvalue, output_range; model_ite = model_ite)
plot!(output_range, Aout_v2, label = "v2 with ite=$model_ite")


######################### test for discrete spectral density

M=4;
poles = - collect(1:M) * output_bound/(M+1)
γ_vec = ones(M)
iwn=(collect(0:(M-1)) .+ 0.5)*2π/β * im;
wn = collect((0:(M-1)) .+ 0.5) * 2π / β;

Gvalue = generate_GFV_delta(β, M, poles, γ_vec; noise = noise);
Aout=my_chi2kink(iwn, Gvalue, output_range)
scatter(
    poles,
    zero(poles),
    title = "noise=$noise",
    xlabel = "poles",
    label = "poles place",
    markersize = 5,
    markercolor = :blue,
    markeralpha = 0.6,
)
plot!(output_range, Aout, label = "reconstruct Spectral")

B = Dict{String,Any}(
    "solver" => "MaxEnt",  # Choose MaxEnt solver
    "mtype" => "gauss",   # Default model function
    "mesh" => "tangent", # Mesh for spectral function
    "ngrid" => M,        # Number of grid points for input data
    "nmesh" => output_number,       # Number of mesh points for output data
    "wmax" => output_bound,       # Right boundary of mesh
    "wmin" => -output_bound,      # Left boundary of mesh
    "beta" => β,      # Inverse temperature
);
S = Dict{String,Any}(
    "nalph" => 15,        # Number of α parameters
    "alpha" => 1e12,      # Starting value of α parameter
    "blur" => -1.0,      # Enable preblur or not
);
setup_param(B, S);
mesh, reA, reG = solve(wn, Gvalue);
plot!(mesh, reA, label = "ACFlow MaxEnt")


model_ite = 100
Aout_v2 = ACFlowSensitivity.my_chi2kink_v2(iwn, Gvalue, output_range; model_ite = model_ite)
plot!(output_range, Aout_v2, label = "v2 with ite=$model_ite")
