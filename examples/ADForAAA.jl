using ACFlowSensitivity
using Plots

N=8;
β=10.0;
# μ=[0.5,-2.5]  .+ 0.1*rand(2); σ=[0.2,0.8] .+ 0.1*rand(2); peak=[1.0,0.3] .+ 0.1*rand(2)  ;
μ=[0.5,-2.5]; σ=[0.2,0.8]; peak=[1.0,0.3];

A=continous_spectral_density(μ,σ,peak);

Giwn=generate_G_values_cont(β,N,A);
wn=(collect(0:N-1).+0.5)*2π/β;
# 记得看一下svd的实现方式，并且设计出符合自己需求的svd 

#------------------------------------------------------------------------
# ForwardAD
solver=Solver("aaa","cont","backward");
aaa=ADaaa(solver,wn,Giwn)[1]
get_loss(wn,Giwn)

#------------------------------------------------------------------------
# Finite Difference
M=10
low_bound=1e-4
up_bound=1e-2
ε=exp.( collect( range( log(low_bound), log(up_bound), M ) ) );
Grad=zeros(M);

for i=1:M
    println(i)
    bbb=aaa_cont_FiniDIff_Chain(wn,Giwn,ε[i])
    Grad[i]=sum(abs.(bbb).^2)^(0.5)
end;

plot(ε,Grad,label="size of grad changes as ε changes")

#--------------------------------------------------------------
# Add noise 
M=10
noise=0.0;
low_bound=1e-4
up_bound=1e-2
ε=exp.( collect( range( log(low_bound), log(up_bound), M ) ) );
G_noise=copy(Giwn);
for i in eachindex(G_noise)
    G_noise[i]+=G_noise[i]*noise*rand()*exp(2π*im*rand())
end;

Grad_noise=zeros(M);


for i=1:M
    println(i)
    bbb=aaa_cont_FiniDIff_Chain(wn,G_noise;ε=ε[i])
    Grad_noise[i]=sum(abs.(bbb).^2)^(0.5)
    # Grad_noise[i]=sum(abs.(bbb))
end;

plot(ε,Grad_noise,label="size of grad changes as ε changes with noise:$noise")

