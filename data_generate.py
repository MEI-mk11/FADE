import numpy as np
import librosa
import os
import random

def add_noise(signal, snr):
    """
    添加噪声到音频信号中以实现指定的信噪比(SNR)
    :param signal: 原始音频信号
    :param snr: 信噪比 (dB)
    :return: 添加噪声后的音频信号
    """
    noise = np.random.randn(len(signal))  # 生成白噪声
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    desired_noise_power = signal_power / (10 ** (snr / 10))
    noise = noise * np.sqrt(desired_noise_power / noise_power)
    return signal + noise

def prepare_data(audio_files, snr_range=(-5, 30), step=3):
    """
    准备训练数据和测试数据
    :param audio_files: 音频文件路径列表
    :param snr_range: 信噪比的范围 (起始信噪比, 终止信噪比)
    :param step: 信噪比步长
    :return: 训练数据和测试数据
    """
    data = {}
    for snr in range(snr_range[0], snr_range[1] + 1, step):
        noisy_audio = []
        for file in audio_files:
            signal, sr = librosa.load(file, sr=None)  # 加载音频文件
            noisy_signal = add_noise(signal, snr)
            noisy_audio.append((noisy_signal, sr))
        data[snr] = noisy_audio
    return data

# 读取音频文件列表
def load_audio_files(audio_dir):
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    return audio_files

# 示例：准备数据
audio_files = load_audio_files("audio_sentences")  # 这里是音频文件的目录路径
train_data = prepare_data(audio_files)