# AD for aaa algorithm on continuous spectral density

function ADaaa(solver::Solver, wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low = -8.0, int_up = 8.0, step = 1e-4)
	if solver.Atype == "cont"
		if solver.Ward == "backward"
			return ADaaa_cont_backward(wn, Giwn; int_low = int_low, int_up = int_up, step = step)
		elseif solver.Ward == "forward"
			return ADaaa_cont_forward(wn, Giwn; int_low = int_low, int_up = int_up, step = step)
		end
	elseif solver.Atype == "delta"
		if solver.Ward == "backward"
			return ADaaa_delta_backward(wn, Giwn; int_low = int_low, int_up = int_up, step = step)
		end
	end
end

#Backward mode

# Barycentric interpolation
struct BarycentricFunction <: Function
	weights::Vector{ComplexF64}
	chosen_iwn::Vector{ComplexF64}
	Giwn::Vector{ComplexF64}
end

function (r::BarycentricFunction)(z::Number)
	w_times_f = r.Giwn .* r.weights
	if isinf(z)
		return sum(w_times_f) / sum(r.weights)
	end
	#
	# Try to determine whether z is a valid node
	k = findfirst(z .== r.chosen_iwn)
	#
	if isnothing(k) # Not at a node
		C = 1 ./ (z .- r.chosen_iwn)
		return sum(C .* w_times_f) / sum(C .* r.weights)
	else            # Interpolation at node
		return r.Giwn[k]
	end
end

# Compute sub Lowner matrix
struct GiwnToL0 <: Function
	iwn::Vector{ComplexF64}
	Index0::Vector{Vector{Int64}}
end

function (f::GiwnToL0)(Giwn::Vector{ComplexF64})
	n = length(f.iwn)
	Lowner = zeros(ComplexF64, n, n)
	for i ∈ 1:n
		for j ∈ 1:n
			if i != j
				Lowner[i, j] = (Giwn[i] - Giwn[j]) / (f.iwn[i] - f.iwn[j])
			end
		end
	end
	return Lowner[f.Index0[1], f.Index0[2]]
end

@adjoint function (f::GiwnToL0)(Giwn::Vector{ComplexF64})
	value = f(Giwn)
	n = length(Giwn)
	m = length(f.Index0[2])
	jac = Matrix{ComplexF64}(undef, m * (n - m), n)
	# L0的 jth列
	for j ∈ 1:m
		# L0的 ith行
		for i ∈ 1:n-m
			# 对Giwn[k]求导
			for k ∈ 1:n
				if k == f.Index0[1][i]
					jac[(j-1)*(n-m)+i, k] = 1 / (f.iwn[f.Index0[1][i]] - f.iwn[f.Index0[2][j]])
				elseif k == f.Index0[2][j]
					jac[(j-1)*(n-m)+i, k] = -1 / (f.iwn[f.Index0[1][i]] - f.iwn[f.Index0[2][j]])
				else
					jac[(j-1)*(n-m)+i, k] = 0
				end
			end
		end
	end
	function pullback(Δ)
		return (nothing, jac' * vec(Δ))
	end

	return value, pullback
end


# Compute Loss function 
struct Loss <: Function
	iwn::Vector{ComplexF64}
	Index0::Vector{Vector{Int64}}
	f0::BarycentricFunction
	int_low::Float64
	int_up::Float64
	step::Float64
	int_field::Vector{Float64}
end

function Loss(iwn::Vector{ComplexF64}, Index0::Vector{Vector{Int64}}, f0::BarycentricFunction, int_low::Float64, int_up::Float64, step::Float64)
	n=(length(int_low:step:int_up)>>1)*2+1 # 为了使用simpson积分，需要奇数个点
	int_field=collect( range(int_low,int_up,n) )
	return Loss(iwn, Index0, f0, int_low, int_up, step, int_field)
end


#= L1 norm version loss function
function (f::Loss)(Giwn::Vector{ComplexF64}, weights::Vector{ComplexF64})
	iwn0 = f.iwn[f.Index0[2]]
	G0 = Giwn[f.Index0[2]]

	int_field = collect(f.int_low:f.step:f.int_up)
	n = length(int_field)
	w_times_f = weights .* G0
	values0 = map(int_field) do z
		C = 1 ./ (z .- iwn0)
		return sum(C .* w_times_f) / sum(C .* weights)
	end

	values = abs.(imag.(values0 - f.f0.(int_field))) / π
	values1 = view(values, 1:n-1)
	values2 = view(values, 2:n)
	return sum((values1 + values2) * f.step / 2)
end
=#


#= L2^2 norm version loss function
function (f::Loss)(Giwn::Vector{ComplexF64}, weights::Vector{ComplexF64})
	iwn0 = f.iwn[f.Index0[2]]
	G0 = Giwn[f.Index0[2]]

	int_field = collect(f.int_low:f.step:f.int_up)
	n = length(int_field)
	w_times_f = weights .* G0
	values0 = map(int_field) do z
		C = 1 ./ (z .- iwn0)
		return sum(C .* w_times_f) / sum(C .* weights)
	end
	# values = abs.( imag.( values0 - f.f0.( int_field ) ) / π )

	values = ( imag.( values0 - f.f0.( int_field ) ) / π ).^2
	values1 = view(values, 1:n-1)
	values2 = view(values, 2:n)
	return sum((values1 + values2) * f.step / 2)
end
=#

# L2 norm version loss function
function (f::Loss)(Giwn::Vector{ComplexF64}, weights::Vector{ComplexF64})
	iwn0 = f.iwn[f.Index0[2]]
	G0 = Giwn[f.Index0[2]]

	w_times_f = weights .* G0
	values0 = map(f.int_field) do z
		C = 1 ./ (z .- iwn0)
		return sum(C .* w_times_f) / sum(C .* weights)
	end
	values = (imag.(values0 - f.f0.(f.int_field)) / π) .^ 2

	# 辛普森法实现积分

	h = f.step  # 步长
	I = values[1] + values[end]  # 边界值
	I += 4 * sum(values[2:2:end-1])  # 奇数项
	I += 2 * sum(values[3:2:end-2])  # 偶数项
	I *= h / 3  # 辛普森公式系数

	result = sqrt(I)
	return result
end
#


# Compute Loss function from Giwn and L0
struct GiwnL0ToLoss <: Function
	iwn::Vector{ComplexF64}
	Index0::Vector{Vector{Int64}}
	f0::BarycentricFunction
	int_low::Float64
	int_up::Float64
	step::Float64
end

function (f::GiwnL0ToLoss)(Giwn::Vector{ComplexF64}, L0::Matrix{ComplexF64})
	weights = svd(L0).V[:, end]
	loss = Loss(f.iwn, f.Index0, f.f0, f.int_low, f.int_up, f.step)
	return loss(Giwn, weights)
end


@adjoint function (f::GiwnL0ToLoss)(Giwn::Vector{ComplexF64}, L0::Matrix{ComplexF64})
	function vague_reci(a::Number)
		# return 1/a
		return a / (a^2 + 1e-8)
	end
	value = f(Giwn, L0)
	loss = Loss(f.iwn, f.Index0, f.f0, f.int_low, f.int_up, f.step)
	U, S, V = svd(L0)
	∂Giwn, ∂weight = Zygote.gradient(loss, Giwn, V[:, end])
	F = zeros(ComplexF64, length(S), length(S))
	# F = Matrix{Float64}(undef, length(S), length(S))
	N = length(S)
	for i ∈ 1:N
		for j ∈ 1:N
			if i != j
				F[i, j] = vague_reci(S[j]^2 - S[i]^2)
			end
		end
	end
	V̄ = zero(V)
	V̄[:, end] = conj(∂weight) / 2
	conj_AK = U * diagm(S) * (F .* (V' * conj(V̄) - transpose(V̄) * V)) * V'
	O = I(N) .* (transpose(V) * V̄)
	reverseS = diagm(vague_reci.(S))
	conj_AO = U * reverseS * (O - O') * V'
	print(norm(conj_AO))
	println()
	return value, Δ -> (nothing, Δ * ∂Giwn, Δ * 2 * conj_AK)
end

# Combine GiwnToL0 and GiwnL0ToLoss to get GiwnToLoss
struct GiwnToLoss <: Function
	f1::GiwnToL0
	f2::GiwnL0ToLoss
end

function (f::GiwnToLoss)(Giwn::Vector{ComplexF64})
	return f.f2(Giwn, f.f1(Giwn))
end


# Perform once aaa algorithm and get constants needed for ADaaa
struct ADaaaBase
	wn::Vector{Float64}
	iwn::Vector{ComplexF64}
	Giwn::Vector{ComplexF64}
	Index0::Vector{Vector{Int64}}
	brcF::BarycentricFunction
end

function ADaaaBase(wn::Vector{Float64}, Giwn::Vector{ComplexF64})
	@assert length(wn) == length(Giwn)
	w, g, v, bi = my_aaa(im * wn, Giwn; isAD = true)
	brcF = BarycentricFunction(w, g, v)
	Index0 = [setdiff(1:length(wn), bi), bi]
	return ADaaaBase(wn, im * wn, Giwn, Index0, brcF)
end

# Main function for applying AD on aaa algorithm
function ADaaa_cont_backward(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low::Float64, int_up::Float64, step::Float64)
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)
	f1 = GiwnToL0(ada.iwn, ada.Index0)
	f2 = GiwnL0ToLoss(ada.iwn, ada.Index0, ada.brcF, int_low, int_up, step)
	f = GiwnToLoss(f1, f2)
	return (Zygote.gradient(f, ada.Giwn)[1], f(ada.Giwn))
end



function get_loss(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low = -5.0, int_up = 5.0, step = 1e-4)
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)

	f1 = GiwnToL0(ada.iwn, ada.Index0)
	f2 = GiwnL0ToLoss(ada.iwn, ada.Index0, ada.brcF, int_low, int_up, step)

	f = GiwnToLoss(f1, f2)
	return f(Giwn)
end


# ----------------------------------------------------------------
# Foeward mode
# To be updated



function ADaaa_cont_forward(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low::Float64, int_up::Float64, step::Float64)
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)
	f1 = GiwnToL0(ada.iwn, ada.Index0)
	f2 = GiwnL0ToLoss(ada.iwn, ada.Index0, ada.brcF, int_low, int_up, step)
	f = GiwnToLoss(f1, f2)
	return ForwardAD(f, ada.Giwn)
end

function ForwardAD(f::GiwnToLoss, Giwn::Vector{ComplexF64})
	∂f1, L0 = ForwardAD(f.f1, Giwn)
	∂f2, loss_value = ForwardAD(f.f2, Giwn, L0)
	∂f = ∂f2[1] + transpose(∂f1[1]) * vec(∂f2[2])
	return ∂f, loss_value
end

function ForwardAD(f::GiwnToL0, Giwn::Vector{ComplexF64})
	L0 = f(Giwn)
	n = length(Giwn)
	m = length(f.Index0[2])
	jac = Matrix{ComplexF64}(undef, m * (n - m), n)
	# L0的 jth列
	count = 0
	for j ∈ 1:m
		# L0的 ith行
		for i ∈ 1:n-m
			count += 1
			# 对Giwn[k]求导
			for k ∈ 1:n
				if k == f.Index0[1][i]
					jac[(j-1)*(n-m)+i, k] = 1 / (f.iwn[f.Index0[1][i]] - f.iwn[f.Index0[2][j]])
				elseif k == f.Index0[2][j]
					jac[(j-1)*(n-m)+i, k] = -1 / (f.iwn[f.Index0[1][i]] - f.iwn[f.Index0[2][j]])
				else
					jac[(j-1)*(n-m)+i, k] = 0
				end
			end
		end
	end

	return (jac,), L0
end

function ForwardAD(f::GiwnL0ToLoss, Giwn::Vector{ComplexF64}, L0::Matrix{ComplexF64})
	loss_value = f(Giwn, L0)
	loss = Loss(f.iwn, f.Index0, f.f0, f.int_low, f.int_up, f.step)
	U, S, V = svd(L0)
	∂Giwn, ∂weight = Zygote.gradient(loss, Giwn, V[:, end])
	F = zeros(ComplexF64, length(S), length(S))
	# F = Matrix{Float64}(undef, length(S), length(S))
	for i ∈ axes(F, 1)
		for j ∈ axes(F, 2)
			if i != j
				F[i, j] = 1 / (S[j]^2 - S[i]^2)
			end
		end
	end
	V̄ = zero(V)
	V̄[:, end] = ∂weight
	K = F .* (transpose(V) * V̄)
	conj_AK = U * diagm(S) * (conj(K) + transpose(K)) * V'
	O = I(length(S)) .* (transpose(V) * V̄)
	conj_AO = U * diagm(1 ./ S) * (O - O') * V' / 2
	return (∂Giwn, 2 * (conj_AO + conj_AK)), loss_value
end











# ----------------------------------------------------------------
# Check the correctness of  backward ADaaa by Finite Difference
# We give up finite difference method because of it's poor numerical stability

function aaa_cont_FiniteDifference_Direct(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low::Float64 = -8.0, int_up::Float64 = 8.0, step::Float64 = 1e-4)
	ε = 1e-14
	n = length(wn)
	e = Vector{Vector{ComplexF64}}(undef, n)
	for i in eachindex(e)
		e[i] = zeros(ComplexF64, n)
		e[i][i] = 1.0 + 0.0im
	end
	_, A0 = reconstruct_spectral_density(im * wn, Giwn)
	_, A1 = reconstruct_spectral_density(im * wn, Giwn + ε * e[1])
	loss_value = quadgk(x -> abs(A0(x) - A1(x)) / ε, int_low, int_up)
	return loss_value
end



# ∂L/∂G =  (∂L/∂w)^T * Jw/JG + (∂L/∂w^*)^T * Jw^*/JG 
function aaa_cont_FiniDIff_Chain(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low::Float64 = -8.0, int_up::Float64 = 8.0, step::Float64 = 1e-4, ε=1e-4)
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)
	f1 = GiwnToL0(ada.iwn, ada.Index0)
	L0 = f1(Giwn)

    # println(ada.Index0)

	f2 = GiwnL0ToLoss(ada.iwn, ada.Index0, ada.brcF, int_low, int_up, step)
	# f = GiwnToLoss(f1, f2)
	loss = Loss(f2.iwn, f2.Index0, f2.f0, f2.int_low, f2.int_up, f2.step)

	_, _, V = svd(L0)
	∇Loss_1G, ∇Loss_w = Zygote.gradient(loss, Giwn, V[:, end])

	∂LossDiv∂w = conj(∇Loss_w) / 2
	∇w_G = zeros(ComplexF64, length(∇Loss_w), length(Giwn))
	JwDivJG = zero(∇w_G)
	e = Vector{Vector{ComplexF64}}(undef, length(Giwn))
	for i in eachindex(e)
		e[i] = zeros(ComplexF64, length(Giwn))
		e[i][i] = 1.0 + 0.0im
	end

	w0 = svd(f1(Giwn)).V[:, end]

	for j ∈ 1:length(Giwn)
		w2 = svd(f1(Giwn + ε * im * e[j])).V[:, end]
		w22 = svd(f1(Giwn - ε * im * e[j])).V[:, end]
		w1 = svd(f1(Giwn + ε * e[j])).V[:, end]
		w11 = svd(f1(Giwn - ε * e[j])).V[:, end]
		if real(dot(w1, w0)) < 0  # 如果两个向量方向相反
			w1 = -w1  # 调整符号
		end
		if real(dot(w11, w0)) < 0
			w11 = -w11
		end
		if real(dot(w2, w0)) < 0
			w2 = -w2
		end
		if real(dot(w22, w0)) < 0
			w22 = -w22
		end
		grad_x = (w1 - w11) / (2 * ε)
		grad_y = (w2 - w22) / (2 * ε)
		∇w_G[:, j] = grad_x + grad_y * im
		JwDivJG[:, j] = (grad_x - grad_y * im) / 2
	end
	∇Loss_2G = (JwDivJG)' * ∇Loss_w + transpose(∇w_G) * ∂LossDiv∂w
	return ∇Loss_2G + ∇Loss_1G
end




