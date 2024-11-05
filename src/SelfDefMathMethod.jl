#Adapted step integral

function Lp_norm(input_f,p::Real, a::Real, b::Real; step::Float64=1e-4)
    f=x->abs(input_f(x))^p
    res=0.0

end


Lp_norm(x->sin(x),1,0,2π)