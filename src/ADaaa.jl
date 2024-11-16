# AD for aaa algorithm on continuous spectral density

# Barycentric interpolation
struct BarycentricFunction <: Function
	iwn::Vector{ComplexF64}
	Giwn::Vector{ComplexF64}
	weights::Vector{ComplexF64}
end

function (r::BarycentricFunction)(z::Number)
	w_times_f = r.Giwn .* r.weights
	if isinf(z)
		return sum(w_times_f) / sum(r.weights)
	end
	#
	# Try to determine whether z is a valid node
	k = findfirst(z .== r.iwn)
	#
	if isnothing(k) # Not at a node
		C = 1 ./ (z .- r.iwn)
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
	n = length(f.Index0)
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
	pullback(Δ) = (nothing, jacobian(f, Giwn)' * vec(Δ))
	return value, pullback
end


# Compute Loss function 
struct Loss <: Function
	iwn::Vector{ComplexF64}
	f0::BarycentricFunction
	int_low::Float64 
	int_up::Float64 
	step::Float64 
end

function (f::Loss)(Giwn::Vector{ComplexF64}, weights::Vector{ComplexF64})
	int_field = collect(int_low:step:int_up)
	n = length(int_field)
	values0 = Vector{ComplexF64}(undef, n)
	w_times_f = weights .* Giwn
	for i = 1:n
		z = int_field[i]
		C = 1 ./ (z .- iwn)
		values0[i] = sum(C .* w_times_f) / sum(C .* weights)
	end
	values = abs.(values0 .- f.f0.(int_field))
	values1 = view(values, 1:n-1)
	values2 = view(values, 2:n)
	return sum((values1 + values2) * f.step / 2)
end


# Compute Loss function from Giwn and L0
struct GiwnL0ToLoss <: Function
	iwn::Vector{ComplexF64}
	f0::BarycentricFunction
	int_low::Float64 
	int_up::Float64 
	step::Float64 
end

function (f::GiwnL0ToLoss)(Giwn::Vector{ComplexF64}, L0::Matrix{ComplexF64})
	weights = svd(L0).Vt[:, end]
	loss = Loss(f.iwn, f.f0, f.int_low, f.int_up, f.step)
	return loss(Giwn, weights)
end

@adjoint function (f::GiwnL0ToLoss)(Giwn::Vector{ComplexF64}, L0::Matrix{ComplexF64})
	value = f(Giwn, L0)
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
	w, g, v, bi = aaa(im * wn, Giwn; isAD = true)
	brcF = BarycentricFunction(w, g, v)
	Index0 = [setdiff(1:length(wn), bi), bi]
	return ADaaaBase(wn, im * wn, Giwn, Index0, brcF)
end

# Main function for applying AD on aaa algorithm
function ADaaa(wn::Vector{Float64}, Giwn::Vector{ComplexF64})
	@assert length(wn) == length(Giwn)
	ada = ADaaaBase(wn, Giwn)


end











# Test adjoint

struct test_func <: Function
	x::Real
	y::Real
end
function (f::test_func)(z::Real)
	return f.x * z + f.y
end

f = test_func(1.0, 2.0)
f(3.0)

my_func3(x) = [x[1]^2, x[2] * x[1]]
jacobian(my_func3, [1.0, 2.0])




#------------
using Zygote
using Zygote: @adjoint
function my_func(A::Matrix{Float64})
	return sum(abs.(A) .^ 2)
end

function my_func1(vec::Vector{Float64})
	return [vec[1] vec[2]; vec[3] vec[1]+vec[2]]
end

gradient(my_func, [1.0 2.0; 3.0 4.0])[1]

jacobian(my_func1, [1.0, 2.0, 3.0])[1]

my_func2(x, y) = x^2 + y^2

@adjoint my_func2(x, y) = my_func2(x, y), Δ -> (2 * Δ[1], 2 * Δ[2])

vec(jacobian(x -> x[1]^2, [1.0, 2.0])[1])

my_func3(x) = x[1]^2 + x[2] * x[1]

@adjoint my_func3(x) = my_func3(x), x -> (2 * x[1] + x[2], x[1])

gradient(my_func3, [1.0, 2.0])[1]


mul(x) = x[1] * x[2]
@adjoint mul(x) = mul(x), y -> (y * [x[2], x[1]], nothing)

gradient(mul, [2.0, 3.0])[1]

mul(x) = x[1] * x[2]
@adjoint mul(x) = mul(x), Δ -> (Δ .* [x[2], x[1]],)

function my_func4(x, y)
	return x[1] + 2 * x[2] + 3 * y[1] + 4 * y[2]
end

gradient(my_func4, [1.0, 2.0], [3.0, 4.0])


function my_func5(x)
	function y(t)
		return x[1] + x[2] * t
	end

	@adjoint my_func5(x) = my_func5(x), Δ -> (Δ .* [0, x[2]],)
	Zygote.refresh()
	return gradient(y, 3.0)[1]
end

using Zygote
using Zygote: @adjoint

struct my_func6 <: Function
	a::Real
	b::Real
end

function (f::my_func6)(t)
	return f.a + f.b * t^2
end

@adjoint my_func6(t) = my_func6(t), Δ -> (Δ * 2*t*my_func6.b,)

@adjoint function (f::my_func6)(t)
	value = f(t)
	pullback(Δ) = (nothing, Δ * 2 * t * f.b)
	return value, pullback
end

gradient(my_func6(1.0, 2.0), 3.0)[1]

my_func6(1.0,2.0)(3.0)

function my_func5(x)
	return my_func6(x)
end


@adjoint f::typeof(my_func5(x::Vector{Float64}))(t) = f(t), Δ -> (Δ .* [0, x[2]],)


my_func6 = my_func5([1.0, 2.0])

gradient(my_func6, 3.0)[1]

struct my_func7 <: Function
    a::Real
    b::Real
    my_func7(a::Real, 
    b::Real=1.0) = new(a, b)
end

function (f::my_func7)(t)
    return f.a + f.b * t
end

mf7 = my_func7(7.0)

mf7(4.0)


function my_func8(a,b)
    return a^2 + b^2
end

gradient(my_func8, 1.0, 2.0)[1]