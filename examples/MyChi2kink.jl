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
noise = 1e-2;
Gvalue = generate_G_values_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
wn = (collect(0:N-1) .+ 0.5) * 2π / β;
iwn = wn * im;

Aout = chi2kink_v2(iwn, Gvalue, output_range)
plot(output_range, A.(output_range), label = "origin Spectral ", title = "noise=$noise")
plot!(output_range, Aout, label = "reconstruct Spectral")
