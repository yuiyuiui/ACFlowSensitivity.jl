include("Method.jl")
using Plots

#计算离散格林函数
# A is the spectral function
# density is the density of poles
# mesh is for the ouput G
function pv(A, density, mesh)

    #创建输入节点
    grid = collect(-log10(density):1/density:log10(density))

    #在输出网格上进行格林函数的计算
    A_value = A.(grid)
    G_value = discrete_GF(grid,A_value,mesh)
    return G_value
end

function viz(A,density, mesh)
    G = pv(A, density, mesh)

    #得到对应的连续的格林函数, benchmark_GF is the ground truth
    limit_G = benchmark_GF(A , last(mesh))

    plot(mesh, G,label="P.V. value of Green Function with poles gap =$(1/density)")
    plot!(limit_G[1],limit_G[2],label="Bnechmark Green Function")
    xlims!(first(mesh), last(mesh))
    ylims!(-10, 10)
end

#construct continous spectral density function
# test function: e^{-x^2}, sigma = 0.5, mean = 0.0, 1.0 is the amplitude
A = continous_spectral_density([0.0],[0.5],[1.0])

oup_b = 2.0

#这里的扰动3/π是必要的，因为最好不要让采样点（输入点）和输出点重合，否则样本点较多时求输出的过程会退化为
#在poles处《直接》求P.V.积分的过程，失去比较的意义（本来就是要看在非pole处是否能演化到pv积分）
mesh = collect(-oup_b:3e-3/π:oup_b)
density=10000

viz(A,density,mesh)