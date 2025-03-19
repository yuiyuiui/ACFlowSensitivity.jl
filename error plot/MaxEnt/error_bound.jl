using ACFlowSensitivity
using Plots,LinearAlgebra,Random

Random.seed!(4)
μ=[0.5,-2.5];σ=[0.2,1.0];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
output_range=range(-output_bound,output_bound,output_number);
output_range=collect(output_range);
iwn=(collect(0:N-1).+0.5)*2π/β * im;
d = output_range[2]-output_range[1]

noise=0.0;
Gvalue=generate_G_values_cont(β,N,A;noise=noise);
Aout = my_chi2kink(iwn,Gvalue,output_range);
dAdivdG, _ = ADchi2kink(iwn,Gvalue,output_range);
norm(dAdivdG)
η = 1e-4

dAout =  η * sum(abs2.(dAdivdG),dims=2) .^ 0.5

Aupper = Aout + 2 * dAout
Alower = max.(Aout-2 * dAout,0.0)


plot(output_range,Aout,title = "noise = $noise and pert = $η",label="origin", xlabel="ω", ylabel="A(ω)")
plot!(output_range, Aupper, fillrange=Alower, fillalpha=0.3, label="Confidence region", linewidth=0)

i = 1

direc = η * rand() * exp.(im * 2π * rand(N))
Aout1 = my_chi2kink(iwn,Gvalue + direc,output_range)
plot!(output_range, Aout1, label="random pert $i", linewidth=0.3)
i+=1

savefig("random_pert.png")


