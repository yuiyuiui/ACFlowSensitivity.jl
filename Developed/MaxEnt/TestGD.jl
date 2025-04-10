using ACFlowSensitivity, Plots, LinearAlgebra, Optim


μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 401;
noise = 0.0;
Gvalue = generate_G_values_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im;



# function of chi2kink
output_number = length(output_range);

# 计算积分时候网格点的权重
d = output_range[2] - output_range[1];
output_weight = fill(d, output_number);

# set the kernel matrix
kernel = Matrix{ComplexF64}(undef, N, output_number);
for i ∈ 1:N
	for j ∈ 1:output_number
		kernel[i, j] = 1 / (iwn[i] - output_range[j])
	end
end;

# real paraliaze Gvalue and kernel
G = vcat(real(Gvalue), imag(Gvalue));
K = [real(kernel); imag(kernel)];
_, S, V = svd(K);
n = count(x -> (x >= 1e-10), S);
V = V[:, 1:n];


# defualt model
model = exp.(-output_range .^ 2 / 2);
# 调整参数，归一化
model = model / (model' * output_weight);

# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
σ = 1e-4;

α = 1e4;


# function Q
A_vec(u::Vector{Float64}) = model .* exp.(V * u)
χ²(u::Vector{Float64}) = (G - d * K * A_vec(u))' * (G - d * K * A_vec(u)) / (σ^2)
Q(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight - 0.5 * χ²(u)

# 𝞉Q/∂u
function ∂Qdiv∂u(u::Vector{Float64}, α::Float64)
    ∂Sdiv∂A = -d * (V * u)'    #行向量	
    ∂χ²div∂A = 2 / (σ^2) * (-d * G' * K + d^2 * A_vec(u)' * K' * K)    #行向量
    ∂Adiv∂u = diagm(A_vec(u)) * V
    ∂Sdiv∂u = ∂Sdiv∂A * ∂Adiv∂u
    ∂χ²div∂u = ∂χ²div∂A * ∂Adiv∂u

    return (α * ∂Sdiv∂u - ∂χ²div∂u / 2)'
end





# 自己写的梯度下降法
function my_GD_v1(f, grad, x0; tol = 1e-3, max_iter = 1000)
	res = copy(x0)
	ite = 0
	while true
		ratio = norm(grad(res))
		# 归一化方向
		direct = -grad(res) / ratio
		if ratio < tol || ite >= max_iter
			println("iterations is $ite")
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
		@show ite, f(res), norm(grad(res)), step
	end
end

u_opt = my_GD_v1(u -> -Q(u, α), u -> -∂Qdiv∂u(u, α), zeros(n))
u_opt = Optim.minimizer(optimize(u -> -Q(u, α), zeros(n), BFGS()))

norm(∂Qdiv∂u(u_opt, α))
-Q(u_opt, α)
grad = ∂Qdiv∂u(u_opt, α)
ratio = norm(grad)
direct = grad / ratio
step = 1.0 / 256
-Q(u_opt + step * direct, α)




using LinearAlgebra
model = rand(401)
my_fun(x) = -sum(exp.(x) - model - exp.(x) .* log.(exp.(x) ./ model)) + sum((exp.(x) .- 1.0) .^ 2)
J(x) = (log.(exp.(x) ./ model) + 2 * (exp.(x) .- 1.0)) .* exp.(x)
opt = my_GD_v1(my_fun, J, rand(401))
norm(J(opt))









#-------=============------------------------ .only for S

μ = [0.5, -2.5];
σ = [0.2, 0.8];
peak = [1.0, 0.3];
A = continous_spectral_density(μ, σ, peak);
β = 10.0;
N = 20;
output_bound = 5.0;
output_number = 401;
noise = 0.0;
Gvalue = generate_G_values_cont(β, N, A; noise = noise);
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
iwn = (collect(0:N-1) .+ 0.5) * 2π / β * im;



# function of chi2kink
output_number = length(output_range);

# 计算积分时候网格点的权重
d = output_range[2] - output_range[1];
output_weight = fill(d, output_number);

# set the kernel matrix
kernel = Matrix{ComplexF64}(undef, N, output_number);
for i ∈ 1:N
	for j ∈ 1:output_number
		kernel[i, j] = 1 / (iwn[i] - output_range[j])
	end
end;

# real paraliaze Gvalue and kernel
G = vcat(real(Gvalue), imag(Gvalue));
K = [real(kernel); imag(kernel)];


# defualt model
model = exp.(-output_range .^ 2 / 2);
# 调整参数，归一化
model = model / (model' * output_weight);

# 默认测量Green function values on image axis时，各个测量值的标准差是1e-4
σ = 1e-4;

α = 1e4;

# function S
A_vec(u::Vector{Float64}) = model .* exp.(u);
S(u::Vector{Float64}, α::Float64) = α * (A_vec(u) - model - A_vec(u) .* log.(A_vec(u) ./ model))' * output_weight;


function ∂Sdiv∂u(u::Vector{Float64}, α::Float64)
	∂Sdiv∂A = -d * u'   #行向量	
	∂Adiv∂u = diagm(A_vec(u))
	return α * (∂Sdiv∂A * ∂Adiv∂u)'
end;

opt = my_GD_v1(u -> -S(u, α), u -> -∂Sdiv∂u(u, α), rand(401))
norm(∂Sdiv∂u(opt, α))



# ----------------------------------------------------------------
# 两种不同的主要的自己写的梯度下降法的比较
# 自己写的梯度下降法
function my_GD_v1(f, grad, x0; tol=1e-6, max_iter=2000)
    res=copy(x0)
    ite=0
    while true
        ratio=norm(grad(res))
        # 归一化方向
        direct=-grad(res)/ratio
        if ratio<tol || ite>=max_iter
            println("iterations is $ite")
            return res
        end

        f_now=f(res)
        step=1.0
        while f(res+direct*step)<f_now-step*ratio*2/3
            step*=2
        end
        while f(res+direct*step)>=f_now-step*ratio/3
            step/=2
        end
        res=res+direct*step
        ite+=1
    end
end


# 自己写的梯度下降法，斜率的角平分线版本，v2效果一般会比v1好一些
function my_GD_v2(f, grad, x0; tol = 1e-6, max_iter = 2000)
	res = copy(x0)
	ite = 0
	while true
		ratio = norm(grad(res))
		# 归一化方向
		direct = -grad(res) / ratio
		if ratio < tol || ite >= max_iter
			println("iterations is $ite")
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
		@show ite, f(res), norm(grad(res)), step
	end
end

using LinearAlgebra
mean=rand(30)
my(x)=sum((x.-mean).^2)
myJ(x)=2*(x.-mean)
guess=rand(30)
u1=my_GD_v1(my,myJ,guess)
u2=my_GD_v2(my,myJ,guess)
norm(grad(u1))
norm(grad(u2))























