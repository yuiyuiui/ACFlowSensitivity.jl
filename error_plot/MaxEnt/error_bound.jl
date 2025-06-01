using ACFlowSensitivity
using LinearAlgebra, Random
using Plots

Random.seed!(1234)
μ=[0.5, -2.5];
σ=[0.2, 1.0];
peak=[1.0, 0.3];
A=continous_spectral_density(μ, σ, peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
output_range=range(-output_bound, output_bound, output_number);
output_range=collect(output_range);
iwn=(collect(0:(N - 1)) .+ 0.5)*2π/β * im;
d = output_range[2]-output_range[1]

noise=1e-3;
Gvalue=generate_GFV_cont(β, N, A; noise=noise);
Aout = my_chi2kink(iwn, Gvalue, output_range);
dAdivdG, _ = ADchi2kink(iwn, Gvalue, output_range);
η = 1e-3

dAout = η * sum(abs2.(dAdivdG); dims=2) .^ 0.5

Aupper = Aout + dAout
Alower = max.(Aout - dAout, 0.0)

plot(output_range,
     Aout;
     title="Error bound, pert = $η",
     label="origin",
     xlabel="ω",
     ylabel="A(ω)")
plot!(output_range,
      Aupper;
      fillrange=Alower,
      fillalpha=0.3,
      label="Confidence region",
      linewidth=0)

i = 1

direc = η * rand() * exp.(im * 2π * rand(N))
Aout1 = my_chi2kink(iwn, Gvalue + direc, output_range)
plot!(output_range, Aout1; label="random pert $i", linewidth=0.3)
i+=1

# ------------------------------------------------------------
using ACFlowSensitivity
using LinearAlgebra, Random
using Plots

Random.seed!(1234)
μ=[0.5, -2.5];
σ=[0.2, 1.0];
peak=[1.0, 0.3];
A=continous_spectral_density(μ, σ, peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
output_range=range(-output_bound, output_bound, output_number);
output_range=collect(output_range);
iwn=(collect(0:(N - 1)) .+ 0.5)*2π/β * im;
d = output_range[2]-output_range[1]

noise=1e-3;
Gvalue=generate_GFV_cont(β, N, A; noise=noise);

T = 10
pert = 10.0 .^ collect(-5.0:0.5:0.0)
ave_norm = zeros(length(pert))
for i in 1:length(pert)
    η = pert[i]
    for j in 1:T
        @show i, j
        direc = η * rand() * exp.(im * 2π * rand(N))
        _, dlossdivdG = my_chi2kink(iwn, Gvalue + direc, output_range)
        ave_norm[i] += norm(dlossdivdG)
    end
    ave_norm[i] /= T
end

plot(pert,
     ave_norm;
     title="Error bound varies with pertubation",
     label="||∇L² loss func||",
     xlabel="pertubation",
     ylabel="average norm")
