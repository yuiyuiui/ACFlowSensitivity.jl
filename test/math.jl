@testset "integral" begin
    for T in [Float32, Float64]
        f = x->x^2
        a = T(0)
        b = T(1)
        h = T(1e-4)
        res = ACFlowSensitivity.integral(f, a, b)
        res1 = ACFlowSensitivity.integral(f, a, b; h=h)
        res_im = ACFlowSensitivity.integral(x->(T(1)+im*T(1))*f(x), a, b)
        @test typeof(res) == T
        @test typeof(res1) == T
        @test typeof(res_im) == Complex{T}
        @test isapprox(res, 1//3, atol=strict_tol(T))
        @test isapprox(res1, 1//3, atol=strict_tol(T))
        @test isapprox(res_im, 1//3 * (1+im), atol=strict_tol(T))
        T != Float64 &&
            @test_throws ErrorException ACFlowSensitivity.integral(x->f(x)+0.0, a, b)
    end
end

@testset "Lp" begin
    for T in [Float32, Float64]
        f = x->cos(x)
        a = T(0)
        b = T(π)
        # x-> abs(cos(x)) is not smooth in [0,π] so it's numerical integral converge much slower and need smaller h
        h = T(1e-6)
        res = ACFlowSensitivity.Lp(f, 2.0, a, b, h=h)
        res_im = ACFlowSensitivity.Lp(x->(1+im)*f(x), 2.0, a, b, h=h)
        @test typeof(res) == T
        @test typeof(res_im) == T
        @test isapprox(res, sqrt(π/2), atol=relax_tol(T))
        @test isapprox(res_im, sqrt(π), atol=relax_tol(T))
        T != Float64 &&
            @test_throws ErrorException ACFlowSensitivity.Lp(x->f(x)+0.0, 2.0, a, b)
    end
end

@testset "Nowton Method" begin
    for T in [Float32, Float64]
        n = 10
        mini_point = rand(T, 10)
        # f(x) = sum((x-mini_point).^2)/2
        J(x) = x - mini_point
        H = diagm(ones(T, n))
        back, _, _ = @constinferred ACFlowSensitivity.newton(x->J(x), x->H, zeros(T, n))
        @test back isa Vector{T}
        @test isapprox(back, mini_point, atol=strict_tol(T))
    end
end

@testset "Curve Fitting" begin
    for T in [Float32, Float64]
        p = rand(T, 4)
        ϕ(x, p) = p[1] .+ p[2] ./ (1 .+ exp.(-p[4] .* (x .- p[3])))
        n = 10
        x = Vector{T}((-n ÷ 2):(n ÷ 2))
        y = ϕ(x, p)
        res = ACFlowSensitivity.curve_fit(ϕ, x, y, ones(T, 4)).param
        @test res isa Vector{T}
        @test isapprox(ϕ(x, res), y, atol=strict_tol(T))
    end
end

@testset "fdgradient" begin
    for T in [Float32, Float64]
        f = x -> [x[1]+x[2], x[1]*x[2]]
        J = x -> [1 1; x[2] x[1]]
        x = rand(T, 2)
        fdres = fdgradient(f, x)
        @test fdres isa Matrix{T}
        @test isapprox(fdres, J(x), atol=strict_tol(T))
        @test jacobian_check_v2v(f, fdres, x)
    end
    for T in [ComplexF32, ComplexF64]
        f = x->[real(x[1])+imag(x[2]), imag(x[1])*real(x[2])]
        J = x->[1 im; im*real(x[2]) imag(x[1])]
        x = rand(T, 2)
        fdres = fdgradient(f, x)
        @test fdres isa Matrix{T}
        @test isapprox(fdres, J(x), atol=strict_tol(T))
        @test jacobian_check_v2v(f, fdres, x)
    end
end

@testset "∇L2loss" begin
    n=10
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        rtol = (real(T) <: Float32) ? 1e-1 : 1e-2
        A = rand(T, 2*n, n)
        f = x->real(A*x)
        x0 = rand(T, n)
        y0 = f(x0)
        w = ones(real(T), 2*n)
        ls = x->loss(f(x), y0, w)
        J = fdgradient(f, x0)
        ∇ls = ACFlowSensitivity.∇L2loss(J, w)
        @test ∇ls[1] isa real(T)
        @test ∇ls[2] isa Vector{T}
        @test isapprox(norm(∇ls[2]), ∇ls[1], atol=strict_tol(real(T)))
        @test gradient_check(ls, ∇ls[2], x0; rtol=rtol)
    end
end

@testset "partial derivative of sigmoid loss function" begin
    n = 10
    ϕ(x, p) = p[1] .+ p[2] ./ (1 .+ exp.(-p[4] .* (x .- p[3])))
    for T in [Float32, Float64]
        x0 = rand(T, n)
        y0 = rand(T, n)
        p0 = rand(T, 4)
        lossϕ = (p, y) -> sum(((x->ϕ(x, p)).(x0)-y) .^ 2)
        dl = (p, y) -> vec(fdgradient(p1->lossϕ(p1, y), p))
        Div∂p² = fdgradient(p->dl(p, y0), p0)
        Div∂p²_expect = ACFlowSensitivity.∂²lossϕDiv∂p²(p0, x0, y0)
        @test Div∂p²_expect isa Matrix{T}
        @show norm(Div∂p² - Div∂p²_expect)
        relax_tol(T)
        @test isapprox(Div∂p², Div∂p²_expect, atol=relax_tol(T), rtol=1e-2)
        Div∂p∂y = fdgradient(y->dl(p0, y), y0)
        Div∂p∂y_expect = ACFlowSensitivity.∂²lossϕDiv∂p∂y(p0, x0, y0)
        @test Div∂p∂y_expect isa Matrix{T}
        @test isapprox(Div∂p∂y, Div∂p∂y_expect, atol=relax_tol(T), rtol=1e-2)
    end
end

@testset "testset" begin
    N = 10
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        x = rand(T, N)
        w = rand(T, N)
        u1 = ACFlowSensitivity.mean(x)
        u2 = ACFlowSensitivity.mean(x, w)
        @test u1 isa T && u2 isa T
        @test isapprox(u1, sum(x)/N, atol=strict_tol(T))
        @test isapprox(u2, sum(x .* w) / sum(w), atol=strict_tol(T))
    end
end
Random.seed!(6)
@testset "Prony approximation" begin
    N = 20
    for T in [Float32, Float64]
        f = x -> 10*exp(-x+im*x) + 20*exp(-2x-im*x+im)
        x = collect(0:(N - 1)) .+ T(0)
        y = f.(x)
        rey1 = PronyApproximation(x, y)(x)
        rey2 = PronyApproximation(x, y, 1e-2)(x)
        @test rey1 isa Vector{Complex{T}}
        @test rey2 isa Vector{Complex{T}}
        @test norm(rey1 - y)/norm(y) < 3e-2
        @test norm(rey2 - y)/norm(y) < 3e-2
    end
end

@testset "find_peaks" begin
    n=2
    for T in [Float32, Float64]
        μ = T(0.5)*rand(T,n) .+ collect(1:n)
        σ = [T(0.01),T(0.01)]
        sort!(μ)
        A, ctx, _ = dfcfg(T;μ=μ,σ=σ)
        v = A.(ctx.mesh)
        d = findmax(ctx.mesh[2:end] - ctx.mesh[1:end-1])[1] # max grid width
        idx = ACFlowSensitivity.find_peaks(v, 0.1)
        @test length(idx) == n
        @test isapprox(ctx.mesh[idx], μ, atol=d*sqrt(2))
    end
end


