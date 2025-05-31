using ACFlow, DelimitedFiles, Plots
import ACFlowSensitivity: generate_GFV_delta


poles = [-0.5, 1.2]
γ = [0.7, 0.3]
β = 10.0;
N = 20;
output_bound = 8.0;
output_number = 801;
noise = 0.0;
wn = collect((0:(N-1)) .+ 0.5) * 2π / β;
GFV = generate_GFV_delta(β, N, poles, γ; noise = noise);

B = Dict{String,Any}(
    "solver" => "StochPX",  # Choose MaxEnt solver
    "mtype" => "gauss",   # Default model function
    "mesh" => "tangent", # Mesh for spectral function
    "ngrid" => N,        # Number of grid points for input data
    "nmesh" => output_number,       # Number of mesh points for output data
    "wmax" => output_bound,       # Right boundary of mesh
    "wmin" => -output_bound,      # Left boundary of mesh
    "beta" => β,      # Inverse temperature
);


S = Dict{String,Any}(
    "method" => "best",
    "nfine" => 100000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
    "npole" => 2,    # Number of poles on the real axis. These poles are used to mimic the Matsubara Green's function.
    "ntry" => 10,   # Number of attempts to figure out the solution.
    "nstep" => 1000000,    #  Number of Monte Carlo sweeping steps per attempt / try.
    "theta" => 1e+6,    # . Artificial inverse temperature θ. When it is increased, the transition probabilities of Monte Carlo updates will decrease.
    "eta" => 1e-4,
);



setup_param(B, S);

mesh, reA, reG = solve(wn, GFV);

plot(mesh, reA)
reG

# ---------

using Optim
Np=length(poles)
function loss(x)
    poles = x[1:Np]
    γ = x[(Np+1):(2*Np)]
    res = 0.0
    for j = 1:N
        res1=GFV[j]
        for k = 1:Np
            res1 -= γ[k]/(im*wn[j]-poles[k])
        end
        res+=abs(res1)^2
    end
    return res
end





result = optimize(loss, [1.0, -1.0, 0.5, 0.5], SimulatedAnnealing())

Optim.minimizer(result)
