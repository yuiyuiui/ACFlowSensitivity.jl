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
