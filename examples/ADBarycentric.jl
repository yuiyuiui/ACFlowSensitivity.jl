using ACFlowSensitivity
using Enzyme
using LinearAlgebra

β=10.0;
N=10;
μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);

reA=AD_aaa(β,N);
G_values=generate_G_values_cont(β,N,A);

Enzyme.autodiff(Reverse, G_values->real(reA(G_values)), Active, Active(G_values));


#------------------
#some exercise
function my_func(X::Vector{Float64})
    return maximum(X)
end

x=rand(100);
bx=zeros(100);
y=[0.0];
by=[1.0];
Enzyme.autodiff(Reverse, my_func, Duplicated(x, bx), Duplicated(y, by));



rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2
rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

autodiff(ForwardWithPrimal, rosenbrock, Duplicated(1.0,1.0), Duplicated(3.0, 1.0))


