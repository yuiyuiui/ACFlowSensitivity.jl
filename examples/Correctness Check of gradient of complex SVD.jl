#-------------------------------------------------
# Correctness Check of gradient of complex SVD

using LinearAlgebra
using Zygote


function loss_A(A::Matrix{ComplexF64})
    w=svd(A).V[:,end]
    return sum(real.(w'*w)).^(0.5)
end;



function loss_w(w::Vector{ComplexF64})
    return sum(real.(w'*w)).^(0.5)
end;


N=20;
A=rand(ComplexF64,N,N);

U,S,V=svd(A);
∇Loss_w=gradient(loss_w,V[:,end])[1];

V̄=zero(V);
#V̄[:,end]=conj(V[:,end])

V̄[:,end]=conj(∇Loss_w)/2;

F=zeros(ComplexF64,N,N);
for i=1:N
    for j=1:N
        if i!=j
            F[i,j]=1/( S[j]^2-S[i]^2 )
        end
    end
end;
conj_AK = U * diagm(S) * ( F.* ( V'*conj(V̄) - transpose(V̄)*V ) )*V';
∇Loss_A=2*conj_AK;

# analytical gradient
ϵ=1e-2;
∇Loss_A_FiniDiff=zeros(ComplexF64,N,N);
for i=1:N
    for j=1:N
        E=zeros(ComplexF64,N,N)
        E[i,j]=1.0;
        ∇Loss_A_FiniDiff[i,j]=( loss_A(A+ϵ*E) - loss_A(A) )/ϵ + im*( loss_A(A+im*ϵ*E ) - loss_A(A) )/ϵ;
    end
end;

∇Loss_A
∇Loss_A_FiniDiff
norm(∇Loss_A-∇Loss_A_FiniDiff)






O=I(length(S)).*(transpose(V)*V̄)
conj_AO=U*diagm(1 ./S)*(O-O')*V'
norm(conj_AO)
