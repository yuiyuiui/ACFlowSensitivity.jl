using Test, ACFlowSensitivity

@testset "Cont_AAA" begin
    μ=[0.5,-2.5];σ=[0.2,0.8];peak=[1.0,0.3];
    A=continous_spectral_density(μ,σ,peak);
    β=10.0;
    N=20;
    output_bound=5.0;
    output_number=801;
    Amesh,reconstruct_A,_=aaa_check(A;β,N,output_bound,output_number)
    res=0.0
    reA_value=reconstruct_A.(Amesh)
    A_value=A.(Amesh)
    for i=1:length(Amesh)-1
        res+=(abs(reA_value[i]-A_value[i])+
        abs(reA_value[i+1]-A_value[i+1]))*
        (Amesh[i+1]-Amesh[i])/2
    end
    @test abs(res)<2*1e-2

end
