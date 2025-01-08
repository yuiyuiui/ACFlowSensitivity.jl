

function my_chi2kink(iwn::Vector{ComplexF64}, Gvalue::Vector{ComplexF64}, output_range::Vector{Float64})
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
	U, S, V = svd(K)
	n = count(x -> (x > 1e-10), S)
	U = U[:, 1:n]
	V = V[:, 1:n]

	# defualt model
	model = exp.(-output_range .^ 2 / 2)
	# 调整参数，归一化
	model = model / (model' * output_weight)

	# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
	σ = 1e-4

	# A, χ²和u的关系
	A(u::Vector{Float64}) = model .* exp.(V * u)
	χ²(u::Vector{Float64}) = (G - K * (A(u) .* output_weight))' * (G - K * (A(u) .* output_weight)) / (σ^2)

	# 设定一列 α, 长度默认为20
	L = 18
	α_vec = Vector{Float64}(undef, L)
	α_vec[1] = 1e12
	for i ∈ 2:L
		α_vec[i] = α_vec[i-1] / 10.0
	end

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
	function fitfun(x, p)
		return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
	end

	guess_fit = ones(4)
	_, _, c, d = curve_fit(fitfun, log10.(α_vec), log10.(χ²_vec), guess_fit).param
	

	# 选取拐点，并为了防止过拟合或者欠拟合做一定处理，再计算对应的u
	α_opt = 10.0^(c - 2.5 / d)
	options = Optim.Options(g_tol = 1e-7, f_tol = 1e-7,iterations=1000)
	u_opt = Optim.minimizer(optimize(u -> -Q(u, α_opt), zeros(n), BFGS(), options))

	#复原返回要求的A
	return A(u_opt)
end

using LsqFit
function fitfun(x, p)
	return @. p[1] + p[2] / (1.0 + exp(-p[4] * (x - p[3])))
end

guess_fit = [0.0, 5.0, 2.0, 0.0]
real_fun(x) = 1.1 + 2.2 / (1.0 + exp(-4.4 * (x - 3.3)))
xxx = collect(range(-10, 10, 20))
yyy = real_fun.(xxx)
_, _, c, d = curve_fit(fitfun, xxx, yyy, guess_fit).param





