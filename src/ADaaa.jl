# AD for aaa algorithm on continuous spectral density

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
	jac = Matrix{ComplexF64}(undef, m*(n - m), n)
	# L0的 jth列
	count=0
	for j ∈ 1:m
		# L0的 ith行
		for i ∈ 1:n-m
			count+=1
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
		return (nothing, transpose(jac) * vec(Δ))
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
end

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
	value = f(Giwn, L0)
	loss = Loss(f.iwn, f.Index0, f.f0, f.int_low, f.int_up, f.step)
	U, S, V = svd(L0)
	∂Giwn, ∂weight = gradient(loss, Giwn, V[:, end])
	F=zeros(ComplexF64,length(S),length(S))
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
	conj_AK = U * diagm(S) * ( conj(K) + transpose(K) ) * V'
	O = I(length(S)) .* (transpose(V) * V̄)
	conj_AO = U * diagm(1 ./ S) * (O - O') * V' / 2
	return value, Δ -> (nothing, Δ * ∂Giwn, 2 * Δ * (conj_AO + conj_AK))
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
function ADaaa(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low = -8.0, int_up = 8.0, step = 1e-4)
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)
	f1 = GiwnToL0(ada.iwn, ada.Index0)
	f2 = GiwnL0ToLoss(ada.iwn, ada.Index0, ada.brcF, int_low, int_up, step)
	f = GiwnToLoss(f1, f2)
	return gradient(f, ada.Giwn)[1]
end

function get_loss(wn::Vector{Float64}, Giwn::Vector{ComplexF64}; int_low = -5.0, int_up = 5.0, step = 1e-4)
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)

	f1 = GiwnToL0(ada.iwn, ada.Index0)
	f2 = GiwnL0ToLoss(ada.iwn, ada.Index0, ada.brcF, int_low, int_up, step)

	f = GiwnToLoss(f1, f2)
	return f(Giwn)
end

