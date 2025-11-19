using ACFlowSensitivity, LinearAlgebra, Random
include("../test/testsetup.jl")

function sst(J::Matrix{T}, spt::SpectrumType) where {T<:Complex}
    N = size(J, 2)
    if spt isa Cont
        J1 = Diagonal(sqrt.(ctx.mesh.weight))*[real(J) imag(J)]
        return svd(J1).S[1]
    elseif spt isa Delta
        J1 = [real(J) imag(J)]
        return svd(J1).S[1]
    else
        error("Unsupported spectrum type")
    end
end

function generate_sst(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::Solver) where {T<:Real}
    if ctx.spt isa Cont
        Aout, J = solvediff(GFV, ctx, alg)
        return sst(J, ctx.spt), cal_chi2(Aout, GFV, ctx), Aout, J
    elseif ctx.spt isa Delta
        Aout, (p, γ), (J, _) = solvediff(GFV, ctx, alg)
        return sst(J, ctx.spt), cal_chi2(p, γ, GFV, ctx), Aout, (p, γ), J
    else
        error("Unsupported spectrum type")
    end
end
