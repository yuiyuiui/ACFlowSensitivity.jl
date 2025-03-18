using ACFlowSensitivity
using Plots,LinearAlgebra,Random

Random.seed!(4)
μ=[0.5,-2.5];σ=[0.2,1.0];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
noise=1e-4;
Gvalue=generate_G_values_cont(β,N,A;noise=noise);
output_range=range(-output_bound,output_bound,output_number);
output_range=collect(output_range);
iwn=(collect(0:N-1).+0.5)*2π/β * im;

Aout=my_chi2kink(iwn,Gvalue,output_range)
plot(output_range,A.(output_range),label="origin Spectral ",title="noise=$noise")
plot!(output_range,Aout,label="reconstruct Spectral")

d = output_range[2]-output_range[1]
η = 1e-8

ADAout = ADchi2kink(iwn,Gvalue,output_range);
@show norm(ADAout)
δ = sum(exp.(my_chi2kink(iwn,Gvalue + η * ADAout, output_range) - Aout))*d - output_number*d
δ1= η *norm(ADAout)^2

norm(δ- δ1)/max(δ,δ1)

ADAout_v2 = ACFlowSensitivity.ADchi2kink_v2(iwn,Gvalue,output_range);
norm(ADAout_v2)
δ2 = sum(exp.(my_chi2kink(iwn,Gvalue + η * ADAout_v2, output_range) - Aout))*d - output_number*d
δ3 = η *norm(ADAout_v2)^2
norm(δ2- δ3)/max(δ2,δ3)
# -----
plot(output_range,A.(output_range),title = "compare diff fit constant,noise = $noise",label="origin spectral")
plot!(output_range,Aout,label="fit_const=2.5")
plot!(output_range,Aout,label="fit_const=2b")


######################### Compare with v2

model_ite = 20
Aout_v2 = ACFlowSensitivity.my_chi2kink_v2(iwn,Gvalue,output_range;model_ite=model_ite)

plot!(output_range,Aout_v2,label="v2 with ite=$model_ite")

output_range[2]-output_range[1]

