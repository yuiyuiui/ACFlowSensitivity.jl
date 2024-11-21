using ACFlowSensitivity

N=20;
β=10.0;
μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);

Giwn=generate_G_values_cont(β,N,A);
wn=(collect(0:N-1).+0.5)*2π/β;
# 记得看一下svd的实现方式，并且设计出符合自己需求的svd 
solver=Solver("aaa","cont","backward");
ADaaa(solver,wn,Giwn)[1]



solver.Ward="forward"
ADaaa(solver,wn,Giwn)[1]

#-------------------------------------------------

using LinearAlgebra
using Zygote

#=
function loss(A::Matrix{ComplexF64})
    w=svd(A).V[:,end]
    return sum(real(w'*w))
end
=#

function loss(w::Vector{ComplexF64})
    return sum(real(w'*w))
end

N=20
A=rand(ComplexF64,N,N)

U,S,V=svd(A)
∂w=gradient(loss,V[:,end])[1]

V̄=zero(V)
#V̄[:,end]=conj(V[:,end])

V̄[:,end]=conj(∂w)/2

F=zeros(ComplexF64,N,N)
for i=1:N
    for j=1:N
        if i!=j
            F[i,j]=1/( S[j]^2-S[i]^2 )
        end
    end
end
conj_AK = U * diagm(S) * ( F.* ( V'*conj(V̄) - transpose(V̄)*V ) )*V'
2*conj_AK

O=I(length(S)).*(transpose(V)*V̄)
conj_AO=U*diagm(1 ./S)*(O-O')*V'
norm(conj_AO)




w=rand(ComplexF64,N)
gradient(w->real(w'*w),w)[1]
2w
