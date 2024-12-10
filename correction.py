import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 假设我们有多个刺激的函数数据，每个刺激的函数都有对应的分贝信噪比（dB SNR）和识别准确率（accuracy）
# 每个函数的数据都存在于字典中，键是刺激的名称，值是一个二元组 (dB SNR, accuracy)
stimuli_data = {
    "stimulus_1": (np.array([0, 5, 10, 15, 20]), np.array([0.3, 0.5, 0.6, 0.7, 0.75])),
    "stimulus_2": (np.array([0, 5, 10, 15, 20]), np.array([0.4, 0.55, 0.65, 0.75, 0.8])),
    "stimulus_3": (np.array([0, 5, 10, 15, 20]), np.array([0.35, 0.5, 0.55, 0.68, 0.78])),
    # 更多刺激数据
}


# 1. 计算每个刺激函数的50%准确率点
def find_50_percent_point(dbs, accuracies):
    # 使用插值方法找到准确率为50%时的dB值
    interp_func = interp1d(accuracies, dbs, kind='linear', fill_value='extrapolate')
    return interp_func(0.5)


# 计算每个刺激的50%准确率点
mid_points = {}
for stimulus, (dbs, accuracies) in stimuli_data.items():
    mid_point = find_50_percent_point(dbs, accuracies)
    mid_points[stimulus] = mid_point

# 2. 计算平均函数：在每个dB SNR下计算平均准确率
# 找到所有刺激的dB SNR的联合范围（假设不同刺激的dB SNR值可能不同）
all_dbs = np.unique(np.concatenate([dbs for dbs, _ in stimuli_data.values()]))
average_accuracies = []

# 对每个dB SNR值，计算所有函数的平均准确率
for db in all_dbs:
    accuracies_at_db = []
    for _, (dbs, accuracies) in stimuli_data.items():
        # 使用插值方法获得当前dB SNR下的准确率
        interp_func = interp1d(dbs, accuracies, kind='linear', fill_value='extrapolate')
        accuracies_at_db.append(interp_func(db))
    average_accuracies.append(np.mean(accuracies_at_db))

# 平均函数的准确率为50%的点
average_mid_point = find_50_percent_point(all_dbs, average_accuracies)

# 3. 计算每个刺激函数的平移距离
shift_distances = {stimulus: mid_point - average_mid_point for stimulus, mid_point in mid_points.items()}

# 4. 平移每个刺激函数的中点
shifted_stimuli_data = {}
for stimulus, (dbs, accuracies) in stimuli_data.items():
    # 计算平移后的dB值
    shift_distance = shift_distances[stimulus]
    shifted_dbs = dbs + shift_distance

    # 保存平移后的函数
    shifted_stimuli_data[stimulus] = (shifted_dbs, accuracies)

# 5. 可视化原始函数和平移后的函数
plt.figure(figsize=(10, 6))

# 绘制原始函数
for stimulus, (dbs, accuracies) in stimuli_data.items():
    plt.plot(dbs, accuracies, label=f"Original {stimulus}")

# 绘制平移后的函数
for stimulus, (shifted_dbs, accuracies) in shifted_stimuli_data.items():
    plt.plot(shifted_dbs, accuracies, '--', label=f"Shifted {stimulus}")

# 绘制平均函数
plt.plot(all_dbs, average_accuracies, 'k-', label="Average Function", linewidth=2)

# 设置图例和标签
plt.xlabel("dB SNR")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Stimuli Functions and Average Function with Shifting")
plt.grid(True)
plt.show()

# 输出平移后的数据
for stimulus, (shifted_dbs, accuracies) in shifted_stimuli_data.items():
    print(f"Shifted {stimulus}:")
    print("dB SNR:", shifted_dbs)
    print("Accuracy:", accuracies)
    print()