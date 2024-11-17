using ACFlowSensitivity, LinearAlgebra, Plots

N=20;
β=10.0;
μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);

Giwn=generate_G_values_cont(β,N,A);
wn=(collect(0:N-1).+0.5)*2π/β;
my_func(wn,Giwn)

ADaaa(wn,Giwn)




