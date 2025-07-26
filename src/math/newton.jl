#=
 Newton Method and curve fitting are borrowed from https://github.com/huangli712/ACFlow
=#

# Newton Method
function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
    resid = nothing
    step = T(1)
    limit = T(1e-4)
    try
        resid = -pinv(J) * f
    catch
        resid = zeros(T, length(feed))
    end
    if any(x -> x > limit, abs.(feed))
        ratio = abs.(resid ./ feed)
        max_ratio = maximum(ratio[abs.(feed) .> limit])
        if max_ratio > T(1)
            step = T(1) / max_ratio
        end
    end
    return feed + step .* resid
end
function newton(fun::Function, grad::Function, guess::Vector{T}; maxiter::Int=20000,
                tol::T=T(1e-4)) where {T<:Real}
    counter = 0
    feed = copy(guess)
    f = fun(feed)
    J = grad(feed)
    back = _apply(feed, f, J)
    reach_tol = false

    while true
        counter = counter + 1
        feed += T(1//2) * (back - feed)

        f = fun(feed)
        J = grad(feed)
        back = _apply(feed, f, J)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum(abs.(back - feed)) < tol
            break
        end
    end

    if counter > maxiter
        println("Tolerance is reached in newton()!")
        @show norm(grad(back))
        reach_tol = true
    end

    return back, counter, reach_tol
end
