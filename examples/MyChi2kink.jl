using ACFlowSensitivity
using Plots

μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);
β=10.0;
N=20;
output_bound=8.0;
output_number=801;
noise=0.0;
Gvalue=generate_G_values_cont(β,N,A;noise=noise);
output_range=range(-output_bound,output_bound,output_number);
output_range=collect(output_range);
iwn=(collect(0:N-1).+0.5)*2π/β * im;

Aout=my_chi2kink(iwn,Gvalue,output_range)

plot(output_range,A.(output_range))
plot!(output_range,Aout)
