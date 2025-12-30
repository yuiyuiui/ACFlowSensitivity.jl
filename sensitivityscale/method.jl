using ACFlowSensitivity, LinearAlgebra, Random
using Distributed
using JSON, JLD2
include("../test/testsetup.jl")

addprocs(8)
@everywhere using ACFlowSensitivity

function sst(J::Matrix{T}, spt::SpectrumType, weight::Vector{S}) where {T<:Complex,S<:Real}
    if spt isa Cont
        J1 = Diagonal(sqrt.(weight)) * [real(J) imag(J)]
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
        return sst(J, ctx.spt, ctx.mesh.weight), cal_chi2(Aout, GFV, ctx), Aout, J
    elseif ctx.spt isa Delta
        Aout, (p, γ), (Jp, Jγ) = solvediff(GFV, ctx, alg)
        return sst(Jp, ctx.spt, ctx.mesh.weight), cal_chi2(p, γ, GFV, ctx), Aout, (p, γ),
               (Jp, Jγ)
    else
        error("Unsupported spectrum type")
    end
end

function record_cont(filename::String, sst_val::T, chi2_val::T, Aout_val::Vector{T},
                     J_val::Matrix{Complex{T}}, Aor_val::Vector{T},
                     GFV_val::Vector{Complex{T}},
                     ctx::CtxData{T}, alg::Solver) where {T<:Real}
    data_res = Dict("sst" => sst_val,
                    "chi2" => chi2_val,
                    "Original Spectrum" => Aor_val,
                    "Aout" => Aout_val,
                    "Jacobian Matrix" => J_val)
    open("$(filename)_result.json", "a") do f
        return JSON.print(f, data_res, 4)
    end
    ctx.mesh_type isa TangentMesh ? mesh_type = "TangentMesh" : mesh_type = "UniformMesh"
    ctx.spt isa Cont ? output_format = "Cont" : output_format = "Delta"
    data_cfg = Dict("GFV" => GFV_val,
                    "β" => ctx.β,
                    "N" => ctx.N,
                    "σ" => ctx.σ,
                    "process number" => ctx.nproc,
                    "mesh_bound" => "[$(ctx.mesh.mesh[1]), $(ctx.mesh.mesh[end])",
                    "mesh_length" => length(ctx.mesh.mesh),
                    "mesh_type" => mesh_type,
                    "output_format" => output_format,
                    "algorithm" => string(alg))
    open("$(filename)_configuration.json", "a") do f
        return JSON.print(f, data_cfg, 4)
    end
end

function record_delta(filename::String, sst_val::T, chi2_val::T, Aout_val::Vector{T},
                      (p_val, γ_val),
                      (Jp_val, Jγ_val), (por_val, γor_val), GFV_val::Vector{Complex{T}},
                      ctx::CtxData{T}, alg::Solver) where {T<:Real}
    data_res = Dict("sst" => sst_val,
                    "chi2" => chi2_val,
                    "Original Poles" => "($por_val, $γor_val)",
                    "poles" => "($p_val, $γ_val)",
                    "Aout" => Aout_val,
                    "Jacobian Matrix" => "($Jp_val, $Jγ_val)")
    open("$(filename)_result.json", "a") do f
        return JSON.print(f, data_res, 4)
    end
    ctx.spt isa Delta ? output_format = "Delta" : output_format = "Cont"
    ctx.mesh_type isa TangentMesh ? mesh_type = "TangentMesh" : mesh_type = "UniformMesh"
    data_cfg = Dict("GFV" => GFV_val,
                    "β" => ctx.β,
                    "N" => ctx.N,
                    "σ" => ctx.σ,
                    "find peaks window width" => ctx.fp_ww,
                    "find peaks minimum peak height" => ctx.fp_mp,
                    "process number" => ctx.nproc,
                    "mesh_bound" => "[$(ctx.mesh.mesh[1]), $(ctx.mesh.mesh[end])",
                    "mesh_length" => length(ctx.mesh.mesh),
                    "mesh_type" => mesh_type,
                    "output_format" => output_format,
                    "algorithm" => string(alg))
    open("$(filename)_configuration.json", "a") do f
        return JSON.print(f, data_cfg, 4)
    end
end
