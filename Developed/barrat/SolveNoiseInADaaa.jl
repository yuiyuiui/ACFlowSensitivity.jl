using ACFlowSensitivity
using Plots, LinearAlgebra, Zygote

# it seems that only when N=20, finite difference works

N=20;
β=10.0;
μ=[0.5, -2.5];
σ=[0.2, 0.8];
peak=[1.0, 0.3];
A=continous_spectral_density(μ, σ, peak);
noise=0.0

Giwn=generate_GFV_cont(β, N, A);
for i in eachindex(Giwn)
    Giwn[i]+=Giwn[i]*noise*rand()*exp(2π*im*rand())
end;
wn=(collect(0:(N - 1)) .+ 0.5)*2π/β;
ε=exp.(collect(range(log(1e-4), log(1e-2), 20)));
abs_grad=zeros(20);

function get_abs_grad(wn, Giwn, ε)
    ada = ADaaaBase(wn, Giwn)
    f1 = GiwnToL0(ada.iwn, ada.Index0)
    S=svd(f1(Giwn)).S
    # @show S[1]/S[end]

    w0 = svd(f1(Giwn)).V[:, end]
    ∇w_G = zeros(ComplexF64, length(w0), length(Giwn))
    e = Vector{Vector{ComplexF64}}(undef, length(Giwn))
    for i in eachindex(e)
        e[i] = zeros(ComplexF64, length(Giwn))
        e[i][i] = 1.0 + 0.0im
    end

    for j in 1:length(Giwn)
        w2 = svd(f1(Giwn + ε * im * e[j])).V[:, end]
        w22 = svd(f1(Giwn - ε * im * e[j])).V[:, end]
        w1 = svd(f1(Giwn + ε * e[j])).V[:, end]
        w11 = svd(f1(Giwn - ε * e[j])).V[:, end]
        if real(dot(w1, w0)) < 0  # 如果两个向量方向相反
            w1 = -w1  # 调整符号
        end
        if real(dot(w11, w0)) < 0
            w11 = -w11
        end
        if real(dot(w2, w0)) < 0
            w2 = -w2
        end
        if real(dot(w22, w0)) < 0
            w22 = -w22
        end
        grad_x = (w1 - w11) / (2 * ε)
        grad_y = (w2 - w22) / (2 * ε)
        ∇w_G[:, j] = grad_x + grad_y * im
    end

    return sum(abs.(∇w_G) .^ 2) .^ (0.5), f1(Giwn)
end;

#------------------------------------------------------------------------------------------------------------------------

_, L0=get_abs_grad(wn, Giwn, 1e-6);

loss_L0(L)=sum(abs.(svd(L).V[:, end]));
loss_L0(L0)

for i in 1:20
    L01=L0+ε[i]*(zero(L0) .+ 1.0im)

    abs_grad[i]=abs((loss_L0(L01-L0))/1e-4)
end;

plot(ε, abs_grad)

# ------------------------------------------------------------------------------------------------------------------------

for i in 1:20
    abs_grad[i], _=get_abs_grad(wn, Giwn, ε[i])
end

plot(ε, abs_grad; label="size of grad changes as ε changes with noise:$noise")
