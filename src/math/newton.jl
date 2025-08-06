"""
    newton(
        fun::Function,
        guess,
        kwargs...;
        maxiter::I64 = 20000,
        mixing::F64 = 0.5
    )

It implements the well-known newton algorithm to locate root of a given
polynomial function. Here, `fun` means the function, `guess` is the initial
solution, and `kwargs...` denotes the required arguments for `fun`. Please
be careful, `func` is a multiple variable function. It not only returns
the value, but also the jacobian matrix of the function.

### Arguments
See above explanations.

### Returns
* sol -> Solution.
* call -> Counter for function call to `fun()`.

See also: [`secant`](@ref).
"""
function newton(
    fun::Function,
    guess::Vector{T},
    kwargs...;
    maxiter::Int = 20000,
    mixing::T = T(0.5)
    ) where {T<:Real}
    function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = T(1)
        limit = T(1e-4)
        try
            resid = - pinv(J) * f
        catch
            resid = zeros(T, length(feed))
        end
        if any(x -> x > limit, abs.(feed))
            ratio = abs.(resid ./ feed)
            max_ratio = maximum( ratio[ abs.(feed) .> limit ] )
            if max_ratio > T(1)
                step = T(1) / max_ratio
            end
        end
        return feed + step .* resid
    end

    counter = 0
    feeds = T[]
    backs = T[]
    reach_tol = false

    f, J = fun(guess, kwargs...)
    back = _apply(guess, f, J)
    push!(feeds, guess)
    push!(backs, back)

    while true
        counter = counter + 1
        feed = feeds[end] + mixing * (backs[end] - feeds[end])

        f, J = fun(feed, kwargs...)
        back = _apply(feed, f, J)
        push!(feeds, feed)
        push!(backs, back)

        any(isnan.(back)) && error("Got NaN!")
        if counter > maxiter || maximum( abs.(back - feed) ) < T(1e-4)
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
