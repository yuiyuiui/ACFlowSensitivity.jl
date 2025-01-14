# 一些说明：
# 1. 为了方便计算，我们统一使用均匀的输出网格点，并记个点间距为d

# Directly maximum Q
function chi2kink_v1(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64}; singular_space = true)
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

	# defualt model
	model = exp.(-output_range .^ 2 / 2)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 长度默认为20
	L = 18
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end

	# 后面log10(α)和log10(χ²)要拟合的曲线
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	# 拟合曲线时候为了防止过拟合设置的参数
	adjust = 2.5

	if singular_space

		_, S, V = svd(K)
		n = count(x -> (x >= 1e-10), S)
		V = V[:, 1:n]

		# A, χ²和u的关系
		A(u::Vector{Float64}) = model .* exp.(V * u)
		χ²(u::Vector{Float64}) = (G - K * (A(u) .* output_weight))' * (G - K * (A(u) .* output_weight)) / (σ^2)

		# 要优化的函数
		function Q(u::Vector{Float64}, α)
			SJEntropy = (A(u) - model - A(u) .* log.(A(u) ./ model))' * output_weight
			return α * SJEntropy - χ²(u) / 2
		end

		# 现在对于每一个α, 寻找对应的最优的u, (tha is to say, A), 并得到此时对应的χ²
		χ²_vec = Vector{Float64}(undef, L)
		options = Optim.Options(g_tol = 1e-5, f_tol = 1e-5)
		for i ∈ 1:L
			@show i
			u_opt = Optim.minimizer(optimize(u -> -Q(u, α_vec[i]), zeros(n), BFGS(), options))
			χ²_vec[i] = χ²(u_opt)
		end


		# 现在进行曲线拟合
		guess_fit = ones(4)
		_, _, c, d = curve_fit(fitfun, log10.(α_vec), log10.(χ²_vec), guess_fit).param


		# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
		α_opt = 10.0^(c - adjust / d)
		u_opt = Optim.minimizer(optimize(u -> -Q(u, α_opt), zeros(n), BFGS()))
		@show χ²(u_opt)

		#复原返回要求的A
		return A(u_opt)

	else
		# this method usually doesn't work because we can't make sure we always have A>0 suring the iteration. 
		# A and χ²
		χ²A(A::Vector{Float64}) = (G - K * (A .* output_weight))' * (G - K * (A .* output_weight)) / (σ^2)

		# 要优化的函数
		QA(A::Vector{Float64}, α) = α * ((A - model - A .* log.(A ./ model))' * output_weight) - χ²A(A) / 2

		χ²_vec = Vector{Float64}(undef, L)
		options = Optim.Options(g_tol = 1e-5, f_tol = 1e-5)
		for i ∈ 1:L
			@show i
			A_opt = Optim.minimizer(optimize(A -> -QA(A, α_vec[i]), ones(output_number), BFGS(), options))
			χ²_vec[i] = χ²(A_opt)
		end


		# 现在进行曲线拟合
		guess_fit = ones(4)
		_, _, c, d = curve_fit(fitfun, log10.(α_vec), log10.(χ²_vec), guess_fit).param


		# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
		α_opt = 10.0^(c - adjust / d)
		A_opt = Optim.minimizer(optimize(u -> -QA(A, α_opt), ones(output_number), BFGS()))
		@show χ²A(A_opt)
		#复原返回要求的A
		return A_opt
	end
end



# wirte optimize max Q by hand for AD
function chi2kink_v2(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64}; singular_space = true)
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
	_, S, V = svd(K)
	n = count(x -> (x >= 1e-10), S)
	V = V[:, 1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 2)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# 设定一列 α, 以及对应的χ², 长度默认
	L = 18
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
	adjust = 2.5

	# function Q
	A(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - d * K * A(u))' * (G - d * K * A(u)) / (σ^2)
	Q(u::Vector{Float64}, α::Float64) = α * (A(u) - model - A(u) .* log.(A(u) ./ model))' * output_weight - 0.5 * χ²(u)

	# 𝞉Q/∂u
	function ∂Qdiv∂u(u::Vector{Float64}, α::Float64)
		∂Sdiv∂A = -d * (V * u)'    #行向量	
		∂χ²div∂A = 2 / (σ^2) * (-d * G' * K + d^2 * A(u)' * K' * K)    #行向量
		∂Adiv∂u = diagm(A(u)) * V
		∂Sdiv∂u = ∂Sdiv∂A * ∂Adiv∂u
		∂χ²div∂u = ∂χ²div∂A * ∂Adiv∂u

		return (α * ∂Sdiv∂u - ∂χ²div∂u / 2)'
	end


	# 接下来用BFGS求最值点
	for i in 1:L
		@show i
		α = α_vec
		u_opt = my_BFGS(u -> -Q(u, α_vec[i]), u -> -∂Qdiv∂u(u, α_vec[i]), zeros(n))
		χ²_vec[i] = χ²(u_opt)
	end

	# 现在进行曲线拟合
	guess_fit = ones(4)
	_, _, c, d = curve_fit(fitfun, log10.(α_vec), log10.(χ²_vec), guess_fit).param


	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c - adjust / d)
	u_opt = my_BFGS(u -> -Q(u, α_opt), u -> -∂Qdiv∂u(u, α_opt), model)
	@show χ²(u_opt)

	#复原返回要求的A
	return A(u_opt)
end







function my_likehood(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64}; singular_space = true)
	output_number = length(output_range)
	N = length(Gvalue)

	# 计算积分时候网格点的权重
	output_weight = zero(output_range)
	output_weight[1] = output_range[2] - output_range[1]
	output_weight[end] = output_range[end] - output_range[end-1]
	output_weight[2:end-1] = (output_range[3:end] - output_range[1:end-2]) / 2

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

	if singular_space
		_, S, V = svd(K)
		n = count(x -> (x >= 1e-12), S)
		V = V[:, 1:n]

		# defualt model
		model = exp.(-output_range .^ 2 / 2)
		# 调整参数，归一化
		model = model / (model' * output_weight)
		#=
		A(u::Vector{Float64}) = model .* exp.(V * u)
		χ²(u::Vector{Float64}) = (G - K * (A(u) .* output_weight))' * (G - K * (A(u) .* output_weight)) *1e10
		=#

		χ²(u::Vector{Float64}) = (G - K * ((V * u) .* output_weight))' * (G - K * ((V * u) .* output_weight)) * 1e12


		u_opt = Optim.minimizer(optimize(u -> χ²(u), zeros(n), BFGS()))
		@show χ²(u_opt)

		return V * u_opt
	else
		χ²A(A::Vector{Float64}) = (G - K * (A .* output_weight))' * (G - K * (A .* output_weight)) * 1e10

		A_opt = Optim.minimizer(optimize(χ²A, zeros(output_number), BFGS()))
		@show χ²A(A_opt)
		return A_opt
	end

end


#=
values0 = map(int_field) do z
		C = 1 ./ (z .- iwn0)
		return sum(C .* w_times_f) / sum(C .* weights)
	end
=#












