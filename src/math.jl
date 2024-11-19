# for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0],[ε],[1/(sqrt(2π)*ε)])
end

# Why we don't use quadgk, but accomplish it by ourselves ?
# Because it takes too much time to calculate the first gradient( parameter->quadgk( func_type(parameter),int_low,int_up )[1], parameter )
# after "using QuadGK"
function Lp_norm(f,p::Real;int_low::Float64=-8.0,int_up::Float64=8.0,step::Float64=1e-4)
    int_field=collect(int_low:step:int_up)
    n=length(int_field)
    values=abs.(f.(int_field)).^p
    values1=view(values,1:n-1)
    values2=view(values,2: n)
    return sum(  ( values1+values2)*step/2  )^(1/p)
end
