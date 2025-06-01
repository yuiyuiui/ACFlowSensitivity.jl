"""
    BarRatContext

Mutable struct. It is used within the BarRat solver only.

### Members
* Gᵥ   -> Input data for correlator.
* grid -> Grid for input data.
* mesh -> Mesh for output spectrum.
* 𝒫    -> Prony approximation for the input data.
* ℬ    -> Barycentric rational function approximation for the input data.
* ℬP   -> It means the positions of the poles.
* ℬA   -> It means the weights / amplitudes of the poles.
"""

mutable struct BarRatContext
    Gᵥ::Vector{C64}
    grid::AbstractGrid
    mesh::AbstractMesh
    𝒫::Union{Missing,PronyApproximation}
    ℬ::Union{Missing,BarycentricFunction}
    ℬP::Vector{C64}
    ℬA::Vector{C64}
end

"""
    BFGSOptimizationResults

It is used to save the optimization results of the BFGS algorithm.
### Members
* x₀         -> Initial guess for the solution.
* minimizer  -> Final results for the solution.
* minimum    -> Objective at the final solution.
* iterations -> Number of iterations.
* δx         -> Absolute change in x.
* Δx         -> Relative change in x.
* δf         -> Absolute change in f.
* Δf         -> Relative change in f.
* resid      -> Maximum gradient of f at the final solution.
* gconv      -> If the convergence criterion is satisfied
"""
mutable struct BFGSOptimizationResults{Tx,Tc,Tf}
    x₀::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    δx::Tc
    Δx::Tc
    δf::Tc
    Δf::Tc
    resid::Tc
    gconv::Bool
end

function solve(S::BarRatSolver, rd::RawData)
    println("[ BarRat ]")
    #
    brc = init(S, rd)
    run(brc)                 # get_r("atype") == "delta" && poles!(brc)
    Aout, Gout = last(brc)
    #
    return brc.mesh.mesh, Aout, Gout
end

#------------------
# code copy
mutable struct my_BarRatContext
    Gv::Vector{C64}
    grid::AbstrctGrid
    mesh::AbstractMesh
    calP::Union{Missing,PronyApproximation}
    calB::Union{Missing,PronyApproximation}
    calBP::Vector{C64}
    calBA::Vector{C64}
end

function poles!(brc)

    #从一组选定的poles的系数去求对应的green function
    function ff(x::Vector{Float64})
        Gn=zeros(ComplexF64, length(brc.Gv))
        iwn=brc.grid.w*im
        for i in eachindex(x)
            Gn=Gn+x[i] ./ (iwn .- brc.calBp[i])
        end
        return sum(abs.(Gn-brc.Gv))
    end

    # 计算ff在权重w处的导数值
    function JJ!(J::Vector{ComplexF64}, x::Vector{ComplexF64})
        return J.=gradient_via_fd(ff, x)
    end

    calP=bc_poles(brc.calB)
    filter!(z->abs(imag(z))<get-r("pcut"), calP)
    @assert length(calP)>0
    brc.calBP=calP

    AA=zeros(ComplexF64, length(calP))
    res=optimize(ff, JJ!, AA; max_iter=500)
    return brc.calBA=res.minimizer
end

#find places of poles
function bc_poles(r::BarycentricFunction)
    w=bc_weights(r)
    z=bc_nodes(r)
    nonzero=@. !iszero(w)
    z, w=z[nonzero], w[nonzero]
    m=length(w)
    B=diagm([zero(Float64); ones(Float64, m)])
    E=[zero(Float64) transpose(w); ones(F64, m) diagm(z)]
    pole=[]
    try
        # eigen of E-λB is poles. It's a wonderful skill. And it also prove 
        # that z_k∈ z must not be a pole becuase det(E-z_kB)=w_k∏_{j≠k}(z_j-z_k),
        # at most differing by a minus sign.
        pole=filter(isfinite, eigvals(E, B))
    catch
        λ=filter(z->abs(z)>1e-13, eigvals(E\B))
        pole=1 ./ λ
    end
    return pole
end

# a genaral primiry method to get derivatives of a complex function
function gradient_via_fd(f, x)
    s=cbrt(eps(Float64))
    ∇f=zero(x)
    xx=copy(x)

    # f=u+iv, df/dz=u'x+iv'x
    @inbounds for i in eachindex(x)
        ϵ=max(s*abs(x[i]), s)

        xxi=x[i]
        xx[i]=xxi+ϵ
        δf=f(xx)
        xx[i]=xxi-ϵ
        δf-=f(xx)
        xx[i]=xxi
        ∇f[i] = real(δf / (2 * ϵ))

        xx[i]=xxi+im*ϵ
        δf=f(xx)
        xx[i]=xxi-im*ϵ
        δf-=f(xx)
        xx[i]=xxi
        ∇f[i] += imag(δf / (2 * ϵ))
        # What we get is u'x+iu'y becuase it's the definition of complex gradient.

    end

    return ∇f
end

# calculate amplitude of poles
function optimize(f, g, x0, max_iter=1000)

    #这个函数里x0只是为了提供类型，实际提供数值在下面的init()函数
    d=BFGSDifferentiable(f, g, x0)

    # 这里不太规范，更改了参数值，但是没有用 "!"
    # 把d初始化，并配上一些优化时候必要的变量
    s=init_state(d, x0)
    iteration=0
    !isfinite(value(d)) || any(!isfinite, gradient(d))

    while !gconv && iteration<max_iter
        iteration+=1

        #检查是否更新成功
        #这里的更新位置实际上用的是牛顿法，也就是x_{k+1}=x_k-H^{-1}(x_k)∇f(x_k)
        ls_success=!update_state!(d, s)
        if !ls_success
            break
        end

        update_g!(d, s)
        update_h!(d, s)

        if !all(isdinite, fradient(d))
            @warn "Terminated early due to NaN in gradient."
            break
        end

        # 检查残差
        gconv(eval_resid(d)<=1e-8)
    end

    return BFGSOptimizationResults(x0,
                                   s.x,
                                   value(d),
                                   iteration,
                                   eval_δx(s),
                                   eval_Δx(s),
                                   eval_δf(d, s),
                                   eval_Δf(d, s),
                                   eval-resid(d),
                                   gconv)
end

function last(brc)
    function pole_green!(_G::Vector{ComplexF64})
        η=get_b("eta")
        @assert η>0
        if η>1
            η-=1
        end
        rA=real(brc.calBA)
        rP=real(brc.calBP)
        for i in eachindex(_G)
            _G[i]=sum(rA ./ (brc.mesh.mesh[i] .- rP+η*im))
        end
    end

    # Reconstruct green function
    _G=brc.calB.(brc.mesh.mesh)
    get_r("atype")=="delta"&&pole_green!(_G)

    Aout=-imag.(_G) ./ π

    return Aout, _G
end
