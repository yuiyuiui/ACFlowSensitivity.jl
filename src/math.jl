# for poles in discrete situation
function kernel(ε::Float64)
    return continous_spectral_density([0.0],[ε],[1/(sqrt(2π)*ε)])
end

function Lp_norm(f,p::Real;int_low::Float64=-8.0,int_up::Float64=8.0)
    integral, _ = quadgk(x -> abs(f(x))^p, int_low, int_up)
    return integral^(1/p)
end
