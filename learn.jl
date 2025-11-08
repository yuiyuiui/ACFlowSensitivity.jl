using CairoMakie

# 创建一个图形窗口
fig = Figure()

# 创建一个坐标轴
ax = Axis(fig[1, 1];
          title="basic example",
          xlabel="x",
          ylabel="y",
          xgridvisible=false,  # 关闭 x 方向网格线
          ygridvisible=false)

# 生成一些数据
x = range(0, 10; length=100)
y = sin.(x)

# 绘制线条
lines!(ax, x, y; color=:blue, linewidth=2, label="sin(x)")

# 绘制散点
scatter!(ax, x[1:10:end], y[1:10:end]; color=:red, markersize=10)

# 添加图例
axislegend(ax)

# 预览图像（类似 Plots.jl 的显示）
display(fig)

# 保存为 PNG
save("basic_example.png", fig)

# 保存为 SVG
save("basic_example.svg", fig)

# 保存为 PDF
save("learn_example.pdf", fig)

println("图形已保存为 learn_example.png, .svg, .pdf")
