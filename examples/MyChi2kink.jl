using ACFlowSensitivity
using Plots,Zygote

μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
A=continous_spectral_density(μ,σ,peak);
β=10.0;
N=20;
output_bound=5.0;
output_number=401;
noise=0.0;
Gvalue=generate_G_values_cont(β,N,A;noise=noise);
output_range=range(-output_bound,output_bound,output_number);
output_range=collect(output_range);
iwn=(collect(0:N-1).+0.5)*2π/β * im;

Aout=my_chi2kink(iwn,Gvalue,output_range)
plot!(output_range,A.(output_range),label="origin Spectral ",title="noise=$noise")
plot(output_range,Aout,label="reconstruct Spectral")

realG = vcat(real(Gvalue),imag(Gvalue))
ADAout = AD_chi2kink(iwn,realG,output_range)
size(ADAout[1])





Aout_1=my_likehood(iwn,Gvalue,output_range)
plot(output_range,A.(output_range),title="noise=$noise, only χ²,singulra space",lable="origin")
plot!(output_range,Aout_1,label="reconstruct")


Aout_1=my_likehood(iwn,Gvalue,output_range; singular_space=false)
plot(output_range,A.(output_range),title="noise=$noise, only χ², direct A",lable="origin")
plot!(output_range,Aout_1,label="reconstruct")





function my(x)
    x+=[1.0,0.0]
    return x
end

my(rand(2))
Zygote.jacobian(my,rand(2))