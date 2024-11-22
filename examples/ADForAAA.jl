using ACFlowSensitivity

N=20;
β=10.0;
# μ=[0.5,-2.5]  .+ 0.1*rand(2); σ=[0.2,0.8] .+ 0.1*rand(2); peak=[1.0,0.3] .+ 0.1*rand(2)  ;
μ=[0.5,-2.5]; σ=[0.2,0.8]; peak=[1.0,0.3];

A=continous_spectral_density(μ,σ,peak);

Giwn=generate_G_values_cont(β,N,A);
wn=(collect(0:N-1).+0.5)*2π/β;
# 记得看一下svd的实现方式，并且设计出符合自己需求的svd 
solver=Solver("aaa","cont","backward");
aaa=ADaaa(solver,wn,Giwn)[1]
get_loss(wn,Giwn)
bbb=aaa_cont_FiniDIff_Chain(wn,Giwn)
sum(abs.(bbb).^2)^(0.5)

solver.Ward="forward"
ADaaa(solver,wn,Giwn)[1]


