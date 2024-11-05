using ACFlowSensitivity
using Plots


μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=801;
Amesh,reconstruct_A,_=aaa_check(A;β,N,output_bound,output_number)

# draw the pictures
plot(Amesh,reconstruct_A.(Amesh),label="reconstruct spectral density")
plot!(Amesh,A.(Amesh),label="origin spectral density")



#------------
#example with a pole 

DA=kernel(0.001)

DAmesh,reconstruct_DA,bary_func=aaa_check(DA;N=20,output_number=801)

plot(DAmesh,reconstruct_DA,label="reconstruct spectral density")
plot!(DAmesh,DA.(DAmesh),label="origin spectral density")

points_to_poles=rand(ComplexF64,5).*[1e-1,1e-2,1e-3,1e-4,1e-5]
res=points_to_poles.*bary_func.(points_to_poles)




#-------------------------
#example with poles
N=10
γ=rand(N)*2;

reγ=DireInverse_check(β,N,γ);



println(γ)
println(reγ)
