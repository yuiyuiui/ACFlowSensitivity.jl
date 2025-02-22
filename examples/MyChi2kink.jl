using ACFlowSensitivity
using Plots,LinearAlgebra

μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);
β=10.0;
N=200;
output_bound=5.0;
output_number=401;
noise=1e-2;
Gvalue=generate_G_values_cont(β,N,A;noise=noise);
output_range=range(-output_bound,output_bound,output_number);
output_range=collect(output_range);
iwn=(collect(0:N-1).+0.5)*2π/β * im;

Aout=my_chi2kink(iwn,Gvalue,output_range)
plot(output_range,A.(output_range),label="origin Spectral ",title="noise=$noise")
plot!(output_range,Aout,label="reconstruct Spectral")

ADAout = ADchi2kink(iwn,Gvalue,output_range)
@show norm(ADAout)

# -----
plot(output_range,A.(output_range),title = "compare diff fit constant,noise = $noise",label="origin spectral")
plot!(output_range,Aout,label="fit_const=2.5")
plot!(output_range,Aout,label="fit_const=2b")


function loss(p,x,y)
    a,b,c,d=p
    s=1 ./ ( 1 .+ exp.(-d*(x.-c))  )
    r=a .+ b*s - y
    return sum(r.^2)
end


function ∂loss_curveDiv∂p(p,x,y)
    a,b,c,d=p
    s=1 ./ ( 1 .+ exp.(-d*(x.-c))  )
    r=a .+ b*s - y
    res = zeros(4)
    res[1]=2*sum(r)
    res[2]=2*sum(s.*r)
    res[3]=-2*b*d*sum(s.*(1 .- s).*r)
    res[4]=2*b*sum(s.*(1 .-s).*(x.-c).*r)
    return res
end


function ∂²loss_curveDiv∂p²(p,x,y)
    res = zeros(4, 4)
    a, b, c, d = p
    L = length(x)
    
    # 计算 sigmoid 函数及其相关项
    s = 1 ./ (1 .+ exp.(-d * (x .- c)))
    s1 = s .* (1 .- s)  # s1 = s * (1 - s)
    r = a .+ b * s .- y  # 残差项
    
    # 填充对角元素
    res[1, 1] = 2 * L
    res[2, 2] = 2 * sum(s.^2)
    res[3, 3] = 2 * b^2 * d^2 * sum(s.^2 .* (1 .- s).^2) + 2 * b * d^2 * sum(s1 .* (1 .- 2 * s) .* r)
    res[4, 4] = 2 * sum(b^2 * s.^2 .* (1 .- s).^2 .* (x .- c).^2 + b * (x .- c).^2 .* s1 .* (1 .- 2 * s) .* r)
    
    # 填充非对角元素
    res[1, 2] = res[2, 1] = 2 * sum(s)
    res[1, 3] = res[3, 1] = -2 * b * d * sum(s1)
    res[1, 4] = res[4, 1] = 2 * b * sum(s1 .* (x .- c))
    res[2, 3] = res[3, 2] = -2 * d * sum(s1 .* (b * s .+ r))
    res[2, 4] = res[4, 2] = 2 * sum(s1 .* (x .- c) .* (b * s .+ r))
    res[3, 4] = res[4, 3] = -2 * b * sum(s1 .* (b * d * s1 .* (x .- c) .+ (1 .+ d * (x .- c) .* (1 .- 2 * s)) .* r))
    
    return res
end

