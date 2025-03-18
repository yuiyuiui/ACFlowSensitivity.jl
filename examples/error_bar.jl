using Plots
using Statistics

# 假设的噪声尺度和误差数据
noise_scales = [0.01, 0.1, 1.0, 10.0]  # 不同噪声尺度
errors = [
    [0.005, 0.006, 0.004, 0.007],  # 噪声尺度 0.01 的误差
    [0.05, 0.06, 0.04, 0.07],      # 噪声尺度 0.1 的误差
    [0.5, 0.6, 0.4, 0.7],          # 噪声尺度 1.0 的误差
    [5.0, 6.0, 4.0, 7.0]           # 噪声尺度 10.0 的误差
]

# 计算每个噪声尺度的误差均值和标准差
error_means = mean.(errors)
error_stds = std.(errors)

# 绘制误差条图
plot(noise_scales, error_means, yerr=error_stds, 
     xlabel="Noise Scale", ylabel="Error", 
     label="Error between grad_f and grad_f_fd",
     title="Error Bar Plot for Gradient Comparison",
     marker=:circle, lw=2, legend=:topleft)

# 保存图像
savefig("error_bar_plot.png")