import scipy.interpolate


def calculate_srt(rrm_matrix, snr_values):
    """
    通过线性插值计算SRT
    :param rrm_matrix: Recognition results map (RRM) 矩阵
    :param snr_values: 信噪比的值
    :return: SRTs
    """
    # 创建插值函数
    f = scipy.interpolate.interp2d(snr_values, snr_values, rrm_matrix, kind='linear', fill_value=0)

    srt_values = []
    for snr in snr_values:
        srt = f(snr, snr)  # 获取依赖于训练SNR的SRT
        srt_values.append(srt)

    return srt_values


# 计算SRTs
srt_values = calculate_srt(rrm_matrix, snr_values)