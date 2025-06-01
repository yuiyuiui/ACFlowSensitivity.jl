using ACFlowSensitivity
using Plots, LinearAlgebra, Random

Random.seed!(6)
μ=[0.5, -2.5];
σ=[0.2, 1.0];
peak=[1.0, 0.3];
A=continous_spectral_density(μ, σ, peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
noise0 = 0.0;
noise1 = 1e-5;
noise2 = 1e-4;
noise3 = 1e-3;

Gvalue0=generate_GFV_cont(β, N, A; noise=noise0);
Gvalue1=generate_GFV_cont(β, N, A; noise=noise1);
Gvalue2=generate_GFV_cont(β, N, A; noise=noise2);
Gvalue3=generate_GFV_cont(β, N, A; noise=noise3);
output_range=range(-output_bound, output_bound, output_number);
output_range=collect(output_range);
iwn=(collect(0:(N - 1)) .+ 0.5)*2π/β * im;

Aout0=my_chi2kink(iwn, Gvalue0, output_range)
Aout1=my_chi2kink(iwn, Gvalue1, output_range)
Aout2=my_chi2kink(iwn, Gvalue2, output_range)
Aout3=my_chi2kink(iwn, Gvalue3, output_range)
plot(output_range,
     A.(output_range);
     label="origin Spectral ",
     title="MaxEnt for smooth type",
     xlabel="ω",
     ylabel="A(ω)",
     legendfontsize=7)
plot!(output_range, Aout1; label="reconstruct A1(ω), noise:1e-5", linewidth=0.5)
plot!(output_range, Aout2; label="reconstruct A2(ω), noise:1e-4", linewidth=0.5)
plot!(output_range, Aout3; label="reconstruct A3(ω), noise:1e-3", linewidth=0.5)
plot!(output_range, Aout0; label="reconstruct A0(ω), noise:0.0", linewidth=0.6)

# ------------------------------------------------------------
Random.seed!(6)
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
output_range=range(-output_bound, output_bound, output_number);
output_range=collect(output_range);
iwn=(collect(0:(N - 1)) .+ 0.5)*2π/β * im;

poles = [-2.0, -1.0, 1.0, 2.0]
γ_vec = 0.25 * ones(length(poles))

noise0 = 0.0;
noise1 = 1e-5;
noise2 = 1e-4;
noise3 = 1e-3;
Gval0=generate_GFV_delta(β, N, poles, γ_vec; noise=noise0);
Gval1=generate_GFV_delta(β, N, poles, γ_vec; noise=noise1);
Gval2=generate_GFV_delta(β, N, poles, γ_vec; noise=noise2);
Gval3=generate_GFV_delta(β, N, poles, γ_vec; noise=noise3);

Aout0_delta=my_chi2kink(iwn, Gval0, output_range)
Aout1_delta=my_chi2kink(iwn, Gval1, output_range)
Aout2_delta=my_chi2kink(iwn, Gval2, output_range)
Aout3_delta=my_chi2kink(iwn, Gval3, output_range)
plot(output_range,
     Aout0_delta;
     label="reconstruct A0(ω), noise:0.0",
     title="MaxEnt for delta type",
     xlabel="ω",
     ylabel="A(ω)",
     legendfontsize=6)
plot!(output_range, Aout1_delta; label="reconstruct A1(ω), noise:1e-5", linewidth=0.5)
plot!(output_range, Aout2_delta; label="reconstruct A2(ω), noise:1e-4", linewidth=0.5)
plot!(output_range, Aout3_delta; label="reconstruct A3(ω), noise:1e-3", linewidth=0.5)
plot!(poles,
      γ_vec;
      seriestype=:stem,
      linecolor=:blue,
      marker=:circle,
      markersize=3,
      linestyle=:dash,
      label="origin poles")
