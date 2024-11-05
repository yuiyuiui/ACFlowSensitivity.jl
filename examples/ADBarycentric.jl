using ACFlowSensitivity
using Enzyme

β=10.0;
N=10;
μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);

reA=AD_aaa(β,N);
G_values=generate_G_values_cont(β,N,A);

Enzyme.autodiff(Reverse, G_values->real(reA(G_values)), Active, Active(G_values));



