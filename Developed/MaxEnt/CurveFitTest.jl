using ACFlowSensitivity, LinearAlgebra, Plots
L = 18
a, b, c, d = 2.0 * rand(4)
d+=3.0;
ϕ(x) = a + b / (1 + exp(-d * (x - c)))
noise = 0.2
xx = collect(range(-5, 5, L))
yy = ϕ.(xx) + (2 * rand(L) .- 1) * noise * b


fitfun(x::Float64, p::Vector{Float64}) = p[1] + p[2] / (1 + exp(-p[4] * (x - p[3])))

function JfdivJp(x::Float64, p::Vector{Float64})
    _, b, c, d=p
    J2 = 1 / (1 + exp(d*(x-c)))
    J3 = - b*d/(1+exp(-d*(x-c)))^2 * exp(-d*(x-c))
    J4 = b / (1+exp(-d*(x-c)))^2 * exp(-d*(x-c)) * (x-c)
    return [1.0, J2, J3, J4]
end

function my_curve_fit(
    fun::Function,
    JfdivJp::Function,
    xx::Vector{Float64},
    yy::Vector{Float64},
    guess::Vector{Float64},
)
    loss(p) = sum(((x -> fun(x, p)).(xx) - yy) .^ 2)
    J(p) = 2*sum((x -> JfdivJp(x, p)).(xx) .* ((x -> fun(x, p)).(xx) - yy))
    return my_GD_v2(loss, J, guess)
end

p_opt = my_curve_fit(fitfun, JfdivJp, xx, yy, ones(4))


reϕ(x)=p_opt[1]+p_opt[2]/(1+exp(-p_opt[4]*(x-p_opt[3])))












# ------------------------------------------------------------------


loss(p) = sum((p[1] .+ p[2] ./ (1 .+ exp.(-p[4] * (xx .- p[3]))) - yy) .^ 2)

# 公式的正确性已被验证
function J(p)
    @assert length(p) == 4
    a, b, c, d = p
    mid1 = 2 * (a .+ b ./ (1 .+ exp.(-d * (xx .- c))) - yy)
    mid2 = 1 ./ (1 .+ exp.(-d * (xx .- c)))
    mid3 = mid2 .^ 2 * (-b) .* exp.(-d * (xx .- c))
    ∂a = mid1' * ones(L)
    ∂b = mid1' * mid2
    ∂c = mid1' * mid3 * d
    ∂d = mid1' * (mid3 .* (c .- xx))
    return [∂a, ∂b, ∂c, ∂d]
end

p_opt=my_GD_v2(loss, J, ones(4))

reϕ(x)=p_opt[1]+p_opt[2]/(1+exp(-p_opt[4]*(x-p_opt[3])))

px=collect(range(-5.0, 5.0, 801))

plot(px, ϕ.(px))
plot!(px, reϕ.(px))

[a, b, c, d]
p_opt
