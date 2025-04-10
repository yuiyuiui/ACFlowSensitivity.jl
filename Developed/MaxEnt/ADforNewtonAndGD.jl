using Zygote,LinearAlgebra

function my_newton(
	fun::Function,
	grad::Function,
	guess;
	maxiter::Int64 = 20000,
	mixing::Float64 = 0.5,
 )
	function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
		resid = nothing
		step = 1.0
		limit = 1e-4
		try
			resid = -pinv(J) * f
		catch
			resid = zeros(Float64, length(feed))
		end
		if any(x -> x > limit, abs.(feed))
			ratio = abs.(resid ./ feed)
			max_ratio = maximum(ratio[abs.(feed).>limit])
			if max_ratio > 1.0
				step = 1.0 / max_ratio
			end
		end
		return feed + step .* resid
	end

	counter = 0
	feeds = []
	backs = []

	f = fun(guess)
	J = grad(guess)
	back = _apply(guess, f, J)
	push!(feeds, guess)
	push!(backs, back)
	res_feed = nothing

	while true
		counter = counter + 1
		feed = feeds[end] + mixing * (backs[end] - feeds[end])

		f = fun(feed)
		J = grad(feed)
		back = _apply(feed, f, J)
		push!(feeds, feed)
		push!(backs, back)

		any(isnan.(back)) && error("Got NaN!")
		if counter > maxiter || maximum(abs.(back - feed)) < 1.e-4
			res_feed = feed
			break
		end
	end

    if counter > maxiter
        println("Tolerance is reached in newton()!")
        @show norm(grad(back))
    end

	return back, counter
end

# 牛顿法，ACFlow 的步长处理
function ad_newton(
	fun::Function,
	grad::Function,
	guess;
	maxiter::Int64 = 20000,
	mixing::Float64 = 0.5,
 )
	function _apply(feed::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
		resid = nothing
		step = 1.0
		limit = 1e-4
		try
			resid = -pinv(J) * f
		catch
			resid = zeros(Float64, length(feed))
		end
		if any(x -> x > limit, abs.(feed))
			ratio = abs.(resid ./ feed)
			max_ratio = maximum(ratio[abs.(feed).>limit])
			if max_ratio > 1.0
				step = 1.0 / max_ratio
			end
		end
		return feed + step .* resid
	end

	counter = 0
    feed = copy(guess)
	f = fun(feed)
	J = grad(feed)
	back = _apply(feed, f, J)

	while true
		counter = counter + 1
		feed +=  mixing * (back - feed)

		f = fun(feed)
		J = grad(feed)
		back = _apply(feed, f, J)

		any(isnan.(back)) && error("Got NaN!")
		if counter > maxiter || maximum(abs.(back - feed)) < 1.e-4
			break
		end
	end

    if counter > maxiter
        println("Tolerance is reached in newton()!")
        @show norm(grad(back))
    end

	return back, counter
end


my(x)=[2*x[1]+4*x[2],2*x[2]+4*x[1]]
grad_my(x)=[2.0 4.0;4.0 2.0]
my(zeros(2))

ad_newton(my,grad_my,10*rand(2))

Zygote.jacobian(u->ad_newton(my,grad_my,u)[1],ones(2))



function my1(x)
    y=copy(x)
    y=ones(2)
    return y
end
Zygote.jacobian(my1,rand(2))


function my2(x)
    my3(y)=x*y
    return my3(3)
end
my2(2.0)
Zygote.gradient(my2,2.0)

using Zygote,LinearAlgebra
function my_GD_v2(f, grad, x0; tol = 1e-3, max_iter = 2000)
	res = copy(x0)
	ite = 0
	while true
		ratio = norm(grad(res))
		# 归一化方向
		direct = -grad(res) / ratio
		if ratio < tol || ite >= max_iter
			println("Iterations is $ite, ratio is $ratio")
			return res
		end

		f_now = f(res)
		step = 1.0
		mid_div = (sqrt(ratio^2 + 1) - 1) / ratio
		while f(res + direct * step) < f_now - step * mid_div
			step *= 2
		end
		while f(res + direct * step) >= f_now - step * mid_div
			step /= 2
		end
		list = collect(range(0, 2 * step, 20)[2:end])
		min_value = f_now
		for i ∈ 1:19
			if f(res + direct * list[i]) < min_value
				step = list[i]
				min_value = f(res + direct * list[i])
			end
		end


		res = res + direct * step
		ite += 1
	end
end
Zygote.@nograd range
f(x)=sum((x.-1.1).^2)
grad(x)=2*(x.-1.1)
my_GD_v2(f,grad,rand(5))
Zygote.jacobian(x->my_GD_v2(f,grad,x),rand(5))