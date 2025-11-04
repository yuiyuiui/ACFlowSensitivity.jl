using ACFlowSensitivity, LinearAlgebra, Random
include("../test/testsetup.jl")

function sst(J::Matrix{T}, spt::SpectrumType) where {T<:Complex}
    N = size(J, 2)
    if spt isa Cont
        J1 = Diagonal(sqrt.(ctx.mesh.weight))*[real(J) imag(J)]
        return svd(J1).S[1]^2, 2*π^N/factorial(N-1)*tr(J1'*J1)
    elseif spt isa Delta
        J1 = [real(J) imag(J)]
        return svd(J1).S[1]^2, 2*π^N/factorial(N-1)*tr(J1'*J1)
    else
        error("Unsupported spectrum type")
    end
end

function generate_sst(GFV::Vector{Complex{T}}, ctx::CtxData{T}, alg::Solver) where {T<:Real}
    _, J = solvediff(GFV, ctx, alg)
    sst_max, sst_ave = sst(J, ctx.spt)
    return sst_max, sst_ave
end
