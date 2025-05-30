@testset "integral" begin
    for T in [Float32, Float64]
        f = x->x^2
        a = T(0)
        b = T(1)
        h = T(1e-4)
        res = ACFlowSensitivity.integral(f,a,b)
        res1 = ACFlowSensitivity.integral(f,a,b;h=h)
        res_im = ACFlowSensitivity.integral(x->(T(1)+im*T(1))*f(x),a,b)
        @test typeof(res) == T
        @test typeof(res1) == T
        @test typeof(res_im) == Complex{T}
        @test isapprox(res,1//3,atol=strict_tol(T))
        @test isapprox(res1,1//3,atol=strict_tol(T))
        @test isapprox(res_im,1//3 * (1+im),atol=strict_tol(T))
        T != Float64 && @test_throws ErrorException ACFlowSensitivity.integral(x->f(x)+0.0,a,b)
    end
end

@testset "Lp" begin
    for T in [Float32, Float64]
        f = x->cos(x)
        a = T(0)
        b = T(π)
        # x-> abs(cos(x)) is not smooth in [0,π] so it's numerical integral converge much slower and need smaller h
        h = T(1e-6)
        res = ACFlowSensitivity.Lp(f,2.0,a,b,h=h)
        res_im = ACFlowSensitivity.Lp(x->(1+im)*f(x),2.0,a,b,h=h)
        @test typeof(res) == T
        @test typeof(res_im) == T
        @test isapprox(res,sqrt(π/2),atol=relax_tol(T))
        @test isapprox(res_im,sqrt(π),atol=relax_tol(T))
        T != Float64 && @test_throws ErrorException ACFlowSensitivity.Lp(x->f(x)+0.0,2.0,a,b)
    end
end

@testset "Normal" begin
    for T in [Float32, Float64]
        d = ACFlowSensitivity.Normal(T(0),T(1))
        @test typeof(rand(d)) == T
        @test typeof(rand(d,10)) == Vector{T}
        n = 10000
        samples = rand(d,n)
        μ = sum(samples)/n
        σ = sqrt(sum((samples .- μ).^2)/n)
        @test isapprox(μ,d.μ,atol=2e-2)
        @test isapprox(σ,d.σ,atol=2e-2)
    end
end