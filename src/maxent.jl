# 一些说明：
# 1. 为了方便计算，我们统一使用均匀的输出网格点，并记个点间距为d
Zygote.@nograd range

# wirte optimize max Q by hand for AD
function my_chi2kink(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
	output_number = length(output_range)
	N = length(Gvalue)

	# 计算积分时候网格点的权重
	d = output_range[2] - output_range[1]
	output_weight = fill(d, output_number)

	# set the kernel matrix
	kernel = Matrix{ComplexF64}(undef, N, output_number)
	for i ∈ 1:N
		for j ∈ 1:output_number
			kernel[i, j] = 1 / (iwn[i] - output_range[j])
		end
	end

	# real paraliaze Gvalue and kernel
	G = vcat(real(Gvalue), imag(Gvalue))
	K = [real(kernel); imag(kernel)]
	U, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]
	U = U[:, 1:n]
	S = S[1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 4)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 16
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end
	χ²_vec = Vector{Float64}(undef, L)

	# 后面log10(α)和log10(χ²)要拟合的曲线
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	# 拟合曲线时候为了防止过拟合设置的参数
	#adjust = 2.5


	# function Q
	A_vec(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
	Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

	# -𝞉Q/∂A
	J(u::Vector{Float64}, α::Float64) = α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

	# -∂²Q/∂A∂u
	H(u::Vector{Float64}, α::Float64) = α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V



	# 接下来用Newton method求最值点
	u_guess = zeros(n)
	u_opt_vec = Vector{Vector{Float64}}(undef, L)
	for i in 1:L
		@show i
		α = α_vec[i]
		u_opt, call, _ = my_newton(u -> J(u, α), u -> H(u, α), u_guess)
		u_guess = copy(u_opt)
		u_opt_vec[i] = copy(u_opt)
		χ²_vec[i] = χ²(u_opt)
		@show log10(α), log10(χ²_vec[i]), norm(J(u_opt, α)), call
	end
	idx = findall(isfinite, χ²_vec)
	α_vec = α_vec[idx]
	χ²_vec = χ²_vec[idx]
	u_opt_vec = u_opt_vec[idx]

	# 现在进行曲线拟合
	guess_fit = [0.0, 5.0, 2.0, 0.0]
	_, _, c, dd = my_curve_fit(log10.(α_vec), log10.(χ²_vec), guess_fit, Newton())[1]


	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c)
	u_guess = copy(u_opt_vec[findmin(abs.(α_vec .- α_opt))[2]])
	u_opt, = my_newton(u -> J(u, α_opt), u -> H(u, α_opt), u_guess)

	#复原返回要求的A
	return A_vec(u_opt)
end
#


function my_chi2kink_v2(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64}; model_ite = 0)
	output_number = length(output_range)
	N = length(Gvalue)

	# 计算积分时候网格点的权重
	d = output_range[2] - output_range[1]
	output_weight = fill(d, output_number)

	# set the kernel matrix
	kernel = Matrix{ComplexF64}(undef, N, output_number)
	for i ∈ 1:N
		for j ∈ 1:output_number
			kernel[i, j] = 1 / (iwn[i] - output_range[j])
		end
	end

	# real paraliaze Gvalue and kernel
	G = vcat(real(Gvalue), imag(Gvalue))
	K = [real(kernel); imag(kernel)]
	U, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]
	U = U[:, 1:n]
	S = S[1:n]

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 16
	A_res = zeros(N)

	for t ∈ 0:model_ite
		println("model iter =  $t")
		if t == 0
			model = exp.(-output_range .^ 2 / 4)
			model = model / (model' * output_weight)
		else
			model = copy(A_res)
		end

		α_vec = Vector{Float64}(undef, L)
		α_vec[1] = 1e12
		for i ∈ 2:L
			α_vec[i] = α_vec[i-1] / 10.0
		end
		χ²_vec = Vector{Float64}(undef, L)

		# 后面log10(α)和log10(χ²)要拟合的曲线
		function fitfun(x, p)
			return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
		end

		# 拟合曲线时候为了防止过拟合设置的参数
		#adjust = 2.5


		# function Q
		A_vec(u::Vector{Float64}) = model .* exp.(V * u)
		χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
		Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

		# -𝞉Q/∂A
		J(u::Vector{Float64}, α::Float64) = α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

		# -∂²Q/∂A∂u
		H(u::Vector{Float64}, α::Float64) = α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V



		# 接下来用Newton method求最值点
		u_guess = zeros(n)
		u_opt_vec = Vector{Vector{Float64}}(undef, L)
		for i in 1:L
			α = α_vec[i]
			u_opt, call, _ = my_newton(u -> J(u, α), u -> H(u, α), u_guess)
			u_guess = copy(u_opt)
			u_opt_vec[i] = copy(u_opt)
			χ²_vec[i] = χ²(u_opt)
		end
		idx = findall(isfinite, χ²_vec)
		α_vec = α_vec[idx]
		χ²_vec = χ²_vec[idx]
		u_opt_vec = u_opt_vec[idx]

		# 现在进行曲线拟合
		guess_fit = [0.0, 5.0, 2.0, 0.0]
		_, _, c, dd = my_curve_fit(log10.(α_vec), log10.(χ²_vec), guess_fit, Newton())[1]


		# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
		α_opt = 10.0^(c)
		u_guess = copy(u_opt_vec[findmin(abs.(α_vec .- α_opt))[2]])
		u_opt, = my_newton(u -> J(u, α_opt), u -> H(u, α_opt), u_guess)

		#复原返回要求的A
		A_res = A_vec(u_opt)
	end
	return A_res
end


function ADchi2kink(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
	N = length(Gvalue)
	Try_num = 1
	noise = 0.0
	c_grad_opt = zeros(ComplexF64, N)
	min_grad = Inf
	fit_res = false
	for i ∈ 1:Try_num
		Gvalue += Gvalue .* rand(N) * noise .* exp.(2π * im * rand(N))
		c_grad, reach_tol = _ADchi2kink(iwn, Gvalue, output_range)
		@show norm(c_grad)
		if !reach_tol && norm(c_grad) < min_grad
			fit_res = true
			c_grad_opt = copy(c_grad)
			min_grad = norm(c_grad)
		end
	end
	!fit_res && error("No fit sensitivity analysis found, please try it again!")
	return c_grad_opt
end



function _ADchi2kink(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
	output_number = length(output_range)
	N = length(Gvalue)

	# 计算积分时候网格点的权重
	d = output_range[2] - output_range[1]
	output_weight = fill(d, output_number)

	# set the kernel matrix
	kernel = Matrix{ComplexF64}(undef, N, output_number)
	for i ∈ 1:N
		for j ∈ 1:output_number
			kernel[i, j] = 1 / (iwn[i] - output_range[j])
		end
	end

	# real paraliaze Gvalue and kernel
	G = vcat(real(Gvalue), imag(Gvalue))
	K = [real(kernel); imag(kernel)]
	U, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]
	U = U[:, 1:n]
	S = S[1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 4)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 16
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end
	χ²_vec = Vector{Float64}(undef, L)

	# 后面log10(α)和log10(χ²)要拟合的曲线
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	# 拟合曲线时候为了防止过拟合设置的参数
	#adjust = 2.5


	# function Q
	A_vec(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
	Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

	# -𝞉Q/∂A, what we get is a vector, that is to say, column vector
	J(u::Vector{Float64}, α::Float64) = α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

	# -∂²Q/∂A∂u, -∂f/∂u
	H(u::Vector{Float64}, α::Float64) = α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V

	# ∂χ²/∂A, get a row vector
	∂χ²div∂A(u::Vector{Float64}) = Matrix(2 / (σ^2) * (-d * G' * K + d^2 * A_vec(u)' * V * diagm(S .^ 2) * V'))

	# ∂A/∂u 
	∂Adiv∂u(u::Vector{Float64}) = diagm(A_vec(u)) * V

	# ∂f/∂G 
	∂fdiv∂G = -1 / (σ^2) * diagm(S) * U'

	# ∂χ²/∂G, get a row vector
	∂χ²div∂G(u::Vector{Float64}) = Matrix(2 / (σ^2) * (G' - d * A_vec(u)' * K'))

	# dχ²/dG 
	dχ²divdG(u::Vector{Float64}, α::Float64) = -∂χ²div∂A(u) * ∂Adiv∂u(u) * pinv(H(u, α)) * ∂fdiv∂G + ∂χ²div∂G(u)

	∂χ²OPTdiv∂G = Matrix{Float64}(undef, L, 2 * N)

	# 接下来用Newton method求最值点
	u_guess = zeros(n)
	u_opt_vec = Vector{Vector{Float64}}(undef, L)
	for i in 1:L
		α = α_vec[i]
		u_opt, = my_newton(u -> J(u, α), u -> H(u, α), u_guess)
		u_guess = copy(u_opt)
		u_opt_vec[i] = copy(u_opt)
		χ²_vec[i] = χ²(u_opt)
		∂χ²OPTdiv∂G[i, :] = dχ²divdG(u_opt, α)
	end
	idx = findall(isfinite, χ²_vec)
	α_vec = α_vec[idx]
	χ²_vec = χ²_vec[idx]
	u_opt_vec = u_opt_vec[idx]


	# 现在进行曲线拟合
	guess_fit = [0.0, 5.0, 2.0, 0.0]
	param, _, reach_tol = my_curve_fit(log10.(α_vec), log10.(χ²_vec), guess_fit, Newton())
	_, _, c, _ = param


	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c)
	u_guess = copy(u_opt_vec[findmin(abs.(α_vec .- α_opt))[2]])
	u_opt, = my_newton(u -> J(u, α_opt), u -> H(u, α_opt), u_guess)

	#复原返回要求的A
	A_opt = A_vec(u_opt)

	function _loss(χ²_vec1::Vector{Float64})
		_, _, c1, _ = my_curve_fit(log10.(α_vec), log10.(χ²_vec1), guess_fit, Newton())[1]
		α_opt1 = 10.0^(c1)
		u_opt1, = my_newton(u -> J(u, α_opt1), u -> H(u, α_opt1), u_opt)
		A_opt1 = A_vec(u_opt1)
		idx = findall(x -> x > 1e-3, A_opt)
		return sum(abs.(A_opt1[idx] - A_opt[idx]))*d
	end

	dlossdivdχ² = Zygote.gradient(_loss, χ²_vec)[1]
	res = (∂χ²OPTdiv∂G)' * dlossdivdχ²
	return res[1:N] + im * res[N+1:2*N], reach_tol
end


function ADchi2kink_v2(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
	N = length(Gvalue)
	Try_num = 1
	noise = 0.0
	c_grad_opt = zeros(ComplexF64, N)
	min_grad = Inf
	fit_res = false
	for i ∈ 1:Try_num
		Gvalue += Gvalue .* rand(N) * noise .* exp.(2π * im * rand(N))
		c_grad, reach_tol = _ADchi2kink_v2(iwn, Gvalue, output_range)
		@show norm(c_grad)
		if !reach_tol && norm(c_grad) < min_grad
			fit_res = true
			c_grad_opt = copy(c_grad)
			min_grad = norm(c_grad)
		end
	end
	!fit_res && error("No fit sensitivity analysis found, please try it again!")
	return c_grad_opt
end

function _ADchi2kink_v2(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
	output_number = length(output_range)
	N = length(Gvalue)

	# 计算积分时候网格点的权重
	d = output_range[2] - output_range[1]
	output_weight = fill(d, output_number)

	# set the kernel matrix
	kernel = Matrix{ComplexF64}(undef, N, output_number)
	for i ∈ 1:N
		for j ∈ 1:output_number
			kernel[i, j] = 1 / (iwn[i] - output_range[j])
		end
	end

	# real paraliaze Gvalue and kernel
	G = vcat(real(Gvalue), imag(Gvalue))
	K = [real(kernel); imag(kernel)]
	U, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]
	U = U[:, 1:n]
	S = S[1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 4)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 16
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end
	χ²_vec = Vector{Float64}(undef, L)

	# 后面log10(α)和log10(χ²)要拟合的曲线
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	# 拟合曲线时候为了防止过拟合设置的参数
	#adjust = 2.5


	# function Q
	A_vec(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
	Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

	# -𝞉Q/∂A, what we get is a vector, that is to say, column vector
	J(u::Vector{Float64}, α::Float64) = α * u + 1 / (σ^2) * (-diagm(S) * U' * G + d * diagm(S)^2 * V' * A_vec(u))

	# -∂²Q/∂A∂u, ∂f/∂u
	H(u::Vector{Float64}, α::Float64) = α * Matrix(I(n)) + d / (σ^2) * diagm(S)^2 * V' * diagm(A_vec(u)) * V

	# ∂χ²/∂A, get a row vector
	∂χ²div∂A(u::Vector{Float64}) = Matrix(2 / (σ^2) * (-d * G' * K + d^2 * A_vec(u)' * V * diagm(S .^ 2) * V'))

	# ∂A/∂u 
	∂Adiv∂u(u::Vector{Float64}) = diagm(A_vec(u)) * V

	# ∂f/∂G 
	∂fdiv∂G = -1 / (σ^2) * diagm(S) * U'

	# ∂χ²/∂G, get a row vector
	∂χ²div∂G(u::Vector{Float64}) = Matrix(2 / (σ^2) * (G' - d * A_vec(u)' * K'))

	# dχ²/dG 
	dχ²divdG(u::Vector{Float64}, α::Float64) = -∂χ²div∂A(u) * ∂Adiv∂u(u) * pinv(H(u, α)) * ∂fdiv∂G + ∂χ²div∂G(u)

	∂χ²OPTdiv∂G = Matrix{Float64}(undef, L, 2 * N)

	# 接下来用Newton method求最值点
	u_guess = zeros(n)
	u_opt_vec = Vector{Vector{Float64}}(undef, L)
	for i in 1:L
		α = α_vec[i]
		u_opt, = my_newton(u -> J(u, α), u -> H(u, α), u_guess)
		u_guess = copy(u_opt)
		u_opt_vec[i] = copy(u_opt)
		χ²_vec[i] = χ²(u_opt)
		∂χ²OPTdiv∂G[i, :] = dχ²divdG(u_opt, α)
	end
	idx = findall(isfinite, χ²_vec)
	α_vec = α_vec[idx]
	χ²_vec = χ²_vec[idx]
	u_opt_vec = u_opt_vec[idx]


	# 现在进行曲线拟合
	guess_fit = [0.0, 5.0, 2.0, 0.0]
	param, _, reach_tol = my_curve_fit(log10.(α_vec), log10.(χ²_vec), guess_fit, Newton())
	_, _, c, _ = param


	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c)
	u_guess = copy(u_opt_vec[findmin(abs.(α_vec .- α_opt))[2]])
	u_opt, = my_newton(u -> J(u, α_opt), u -> H(u, α_opt), u_guess)

	#复原返回要求的A
	A_opt = A_vec(u_opt)

	function _∂²loss_curveDiv∂p²(p,x,y)
		a, b, c, d = p
		L = length(x)
		
		# 计算 sigmoid 函数及其相关项
		s = 1 ./ (1 .+ exp.(-d * (x .- c)))
		s1 = s .* (1 .- s)  # s1 = s * (1 - s)
		r = a .+ b * s .- y  # 残差项
		
		# 填充对角元素
		Jaa = 2 * L
		Jbb = 2 * sum(s.^2)
		Jcc = 2 * b^2 * d^2 * sum(s.^2 .* (1 .- s).^2) + 2 * b * d^2 * sum(s1 .* (1 .- 2 * s) .* r)
		Jdd = 2 * sum(b^2 * s.^2 .* (1 .- s).^2 .* (x .- c).^2 + b * (x .- c).^2 .* s1 .* (1 .- 2 * s) .* r)
		
		# 填充非对角元素
		Jab  = 2 * sum(s)
		Jac = -2 * b * d * sum(s1)
		Jad = 2 * b * sum(s1 .* (x .- c))
		Jbc = -2 * d * sum(s1 .* (b * s .+ r))
		Jbd = 2 * sum(s1 .* (x .- c) .* (b * s .+ r))
		Jcd = -2 * b * sum(s1 .* (b * d * s1 .* (x .- c) .+ (1 .+ d * (x .- c) .* (1 .- 2 * s)) .* r))
		
		return [Jaa Jab Jac Jad;Jab Jbb Jbc Jbd;Jac Jbc Jcc Jcd;Jad Jbd Jcd Jdd]
	end

	function _∂²loss_curveDiv∂p∂y(p, x, y)
		a, b, c, d = p
		L = length(x)
		
		# 计算 sigmoid 函数及其相关项
		s = 1 ./ (1 .+ exp.(-d * (x .- c)))
		s1 = s .* (1 .- s)  # s1 = s * (1 - s)
		
		# 初始化混合偏导数矩阵
		∂²loss_∂p∂y_matrix = zeros(4, L)
		
		# 填充矩阵
		∂²loss_∂p∂y_matrix[1, :] .= -2  # ∂²loss/∂a∂y_i
		∂²loss_∂p∂y_matrix[2, :] = -2 * s  # ∂²loss/∂b∂y_i
		∂²loss_∂p∂y_matrix[3, :] = 2 * b * d * s1  # ∂²loss/∂c∂y_i
		∂²loss_∂p∂y_matrix[4, :] = -2 * b * s1 .* (x .- c)  # ∂²loss/∂d∂y_i
		
		return ∂²loss_∂p∂y_matrix
	end

	p1 = my_curve_fit(log10.(α_vec), log10.(χ²_vec), guess_fit, Newton())[1]
	c1 = p1[3]
	α_opt1 = 10.0^(c1)
	u_opt1, = my_newton(u -> J(u, α_opt1), u -> H(u, α_opt1), u_opt)
	A_opt1 = A_vec(u_opt1)

	arg = (p1,log10.(α_vec),log10.(χ²_vec))

	dpdivdχ²_vec = - _∂²loss_curveDiv∂p²(arg...) \ _∂²loss_curveDiv∂p∂y(arg...)*diagm(1 ./ (χ²_vec * log(10)))
	dα_opt1divdχ²_vec = log(10) * dpdivdχ²_vec[3,:]
	dα_opt1divdG = dα_opt1divdχ²_vec' * ∂χ²OPTdiv∂G
	du_opt1divdG = -pinv(H(u_opt1, α_opt1)) * (∂fdiv∂G + u_opt1 * dα_opt1divdG)
	dA_opt1divdG = ∂Adiv∂u(u_opt1) * du_opt1divdG

	
	function _loss(A_opt1::Vector)
		idx = findall(x -> x > 1e-6, A_opt)
		return sum(abs.(A_opt1[idx] - A_opt[idx]))*d
	end
	
	#=
	function _loss(A_opt1::Vector)
		idx = findall(x -> x > 1e-6, A_opt)
		return sum(A_opt1[idx])*d
	end
	=#

	# output as a vector
	dlossdivdA_opt1 = Zygote.gradient(_loss, A_opt1)[1]
	res = (dA_opt1divdG)' * dlossdivdA_opt1
	return res[1:N] + im * res[N+1:2*N], reach_tol
end





