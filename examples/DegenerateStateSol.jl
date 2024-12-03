using LinearAlgebra, Plots,Zygote

# loss(A)=sum( abs.( svd(A).V[:,end] ).^2 )^(0.5)

# loss_w(w::Vector{ComplexF64}) = sum(abs.(w) .^ 2)^(0.5)

loss_w(w::Vector{ComplexF64}) = sum( abs.(w) )

# x -> x/( x^2 + ε )
Lorentz_broaden(x::Number, ε::Float64=1e-12 ) = x/( x^2 + ε )

# A -> svd -> loss
function grad_svd_loss(A::Matrix{ComplexF64}, ε::Float64 = 1e-12)
    M,N=size(A)
    @assert M>=N
	U, S, V = svd(A)
	∇Loss_w = gradient(loss_w, V[:, end])[1]

	V̄ = zero(V)

	V̄[:, end] = conj(∇Loss_w) / 2

	F = zeros(ComplexF64, N, N)
	for i ∈ 1:N
		for j ∈ 1:N
			if i != j
				F[i, j] =  Lorentz_broaden(S[j]^2 - S[i]^2,ε)
			end
		end
	end
	conj_AK = U * diagm(S) * (F .* (V' * conj(V̄) - transpose(V̄) * V)) * V'
	∇Loss_A = 2 * conj_AK

    return ∇Loss_A
end


# Finite Difference
# ∇L = ∇_w L^T * ( Jw/JA )^* + ∇_w L^† /2 * ∇_A w
function finite_diff_svd_loss(A::Matrix{ComplexF64},ε::Float64 = 1e-4)
    M,N=size(A)
    @assert M>=N
    w=svd(A).V[:,end]
    ∇L_w = gradient(loss_w, w)[1]

    ∇w_A = Matrix{ComplexF64}(undef,N,M*N)
    JwDivJA = Matrix{ComplexF64}(undef,N,M*N)

    for j ∈ 1:N
        for i ∈ 1:M
            A1=copy(A)
            A11=copy(A)
            A2=copy(A)
            A22=copy(A)
            p=(j-1)*N + i
            A1[i,j]+=ε
            A11[i,j]-=ε
            A2[i,j]+=im*ε
            A22[i,j]-=im*ε

            w1=svd(A1).V[:,end]
            w11=svd(A11).V[:,end]
            w2=svd(A2).V[:,end]
            w22=svd(A22).V[:,end]
            grad_x = (w1 - w11) / (2 * ε)
		    grad_y = (w2 - w22) / (2 * ε)
		    ∇w_A[:, p] = grad_x + grad_y * im
		    JwDivJA[:, p] = (grad_x - grad_y * im) / 2
        end
	end

    ∇L = transpose(∇L_w) * conj(JwDivJA) + ∇L_w'/2 * ∇w_A
    return reshape(∇L,M,N)

end


#---------------------------------
M=10;
N=10;
A=rand(ComplexF64,M,N);

t=rand(ComplexF64,M-1)


∇L1=grad_svd_loss(A,1e-12)

∇L2=finite_diff_svd_loss(A,1e-4)

norm(∇L1-∇L2)
norm(∇L1)



