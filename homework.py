import random
import math

# 生成一个1到100之间的10个随机数的序列
sequence = random.sample(range(1, 101), 100)

# 计算序列的均值
mean = sum(sequence) / len(sequence)

# 假设每次测量的误差为0.1
error = 0.1

# 计算n次测量的误差
n = len(sequence)
total_error = error / math.sqrt(n)

print(f"序列: {sequence}")
print(f"均值: {mean}")
print(f"误差: {total_error}")


# import random
# from scipy import stats
#
# # 生成100个随机整数
# x = random.choices(range(1, 101), k=100)
#
# # 以x的分布生成1万个随机数
# y = random.choices(x, k=10000)
#
# # 进行KS检验
# d, p_value = stats.kstest(y, x)
#
# print(f"KS检验结果: D = {d}, p-value = {p_value}")
#
# if p_value > 0.05:
#     print("接受原假设，认为{x_i}和{y_i}的分布一致。")
# else:
#     print("拒绝原假设，认为{x_i}和{y_i}的分布不一致。")


# import numpy as np
# from scipy import fftpack
# import matplotlib.pyplot as plt
#
# # 设置采样率和时间
# sample_rate = 1000
# time = np.arange(0, 10, 1/sample_rate)
#
# # 生成具有3个频率的正弦曲线
# freq1 = 5
# freq2 = 10
# freq3 = 20
# signal = np.sin(2 * np.pi * freq1 * time) + np.sin(2 * np.pi * freq2 * time) + np.sin(2 * np.pi * freq3 * time)
#
# # 添加随机误差
# noise = np.random.normal(0, 0.5, size=len(time))
# signal = signal + noise
#
# # 进行Fourier谱分析
# fft_signal = fftpack.fft(signal)
# frequencies = fftpack.fftfreq(len(fft_signal), 1/sample_rate)
#
# # 绘制信号和频谱图
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(time, signal)
# ax1.set_title("Signal")
# ax2.plot(frequencies, np.abs(fft_signal))
# ax2.set_title("Spectrum")
# plt.show()