# for poles in discrete situation
function kernel(ε::Float64)
	return continous_spectral_density([0.0], [ε], [1 / (sqrt(2π) * ε)])
end

# Why we don't use quadgk, but accomplish it by ourselves ?
# Because it takes too much time to calculate the first gradient( parameter->quadgk( func_type(parameter),int_low,int_up )[1], parameter )
# after "using QuadGK"
function Lp_norm(f, p::Real; int_low::Float64 = -8.0, int_up::Float64 = 8.0, step::Float64 = 1e-4)
	int_field = collect(int_low:step:int_up)
	n = length(int_field)
	values = abs.(f.(int_field)) .^ p
	values1 = view(values, 1:n-1)
	values2 = view(values, 2:n)
	return sum((values1 + values2) * step / 2)^(1 / p)
end


function my_BFGS(f, grad, x0; tol = 1e-6, max_iter = 2000)
	x = x0
	n = length(x)
	H = I(n)  # 初始Hessian矩阵的逆近似

	for iter in 1:max_iter
		g = grad(x)
		if norm(g) < tol
			println("Converged in $iter iterations")
			return x
		end

		p = -H * g
		α = linesearch(f, grad, x, p)

		s = α * p
		x_new = x + s
		g_new = grad(x_new)

		y = g_new - g
		rho = 1 / dot(y, s)
		H = (I(n) - rho * s * y') * H * (I(n) - rho * y * s') + rho * s * s'

		x = x_new
	end

	println("Maximum iterations reached")
	return x
end

function linesearch(f, grad, x, p; c1 = 1e-4, c2 = 0.9, max_iter = 20)
	α = 1.0
	for iter in 1:max_iter
		if f(x + α * p) <= f(x) + c1 * α * dot(grad(x), p) &&
		   dot(grad(x + α * p), p) >= c2 * dot(grad(x), p)
			return α
		end
		α /= 2
	end
	return α
end

# 自己写的梯度下降法
function my_GD_v1(f, grad, x0; tol = 1e-6, max_iter = 2000)
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
		while f(res + direct * step) < f_now - step * ratio * 2 / 3
			step *= 2
		end
		while f(res + direct * step) >= f_now - step * ratio / 3
			step /= 2
		end
		res = res + direct * step
		ite += 1
	end
end


# 自己写的梯度下降法，斜率的角平分线版本，v2效果一般会比v1好一些
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


# 牛顿法，ACFlow 的步长处理
function my_newton(
	fun::Function,
	grad::Function,
	guess;
    print_out = false,
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
        if print_out && counter<=100
            @show norm(feed)
        end
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




# ϕ(x;a,b,c,d)=a+b/(1+exp(-d(x-c))) 的曲线拟合
function my_curve_fit(xx::Vector{Float64}, yy::Vector{Float64}, guess::Vector{Float64})
    @assert length(xx)==length(yy)
	loss(p) = sum((p[1] .+ p[2] ./ (1 .+ exp.(-p[4] * (xx .- p[3]))) - yy) .^ 2)
    L=length(xx)

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

	return my_GD_v2(loss, J, guess)
end
