import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .analyze import upsample_data

def generate_t1(
    pulse_bandpass_filtered, valley_indexes,
    amplitude_list, acceptable_idx_list, resampling_rate):
    """
    選別された波形に対して、正規化・アップサンプリング・平均化を行い、
    平均波形 t1・アップサンプリング波形・プロット用元波形を返す。
    """

    acceptable_pulse_num = len(acceptable_idx_list)
    print(f"Number of acceptable pulses/all pulses: {acceptable_pulse_num}/{len(valley_indexes) - 1}")

    if acceptable_pulse_num == 0:
        return None, None, None, False  # スキップする条件用にFalseを返す

    pulse_waveform_upsampled_list = np.empty((acceptable_pulse_num, resampling_rate))

    max_length = 0
    # まず max_length を測定しながらアップサンプリング
    for i, idx in enumerate(acceptable_idx_list):
        pulse_waveform = pulse_bandpass_filtered[valley_indexes[idx]:valley_indexes[idx + 1]]
        pulse_waveform /= amplitude_list[idx]  # 振幅正規化

        pulse_waveform_upsampled = upsample_data(pulse_waveform, resampling_rate)
        pulse_waveform_upsampled_list[i] = pulse_waveform_upsampled

        max_length = max(max_length, len(pulse_waveform))

    # プロット用原波形の初期化と0埋め
    pulse_waveform_original_list = np.empty((acceptable_pulse_num, max_length))
    mean_for_plot_t1 = np.zeros(max_length)

    for i, idx in enumerate(acceptable_idx_list):
        pulse_waveform = pulse_bandpass_filtered[valley_indexes[idx]:valley_indexes[idx + 1]]
        pulse_waveform_padded = np.pad(pulse_waveform, (0, max_length - len(pulse_waveform)), mode="constant")

        pulse_waveform_original_list[i] = pulse_waveform_padded
        mean_for_plot_t1 += pulse_waveform_padded

    mean_for_plot_t1 /= acceptable_pulse_num
    mean_for_plot_t1 /= max(mean_for_plot_t1)
    
    #　可視化用関数
    # plt.plot(mean_for_plot_t1, color="orange", lw=3)
    # plt.xlabel("Frame")
    # plt.ylabel("Amplitude")
    # plt.title("Average Pulse (t1)")
    # plt.tick_params(labelsize=12)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # 最終平均波形 t1
    t1 = np.mean(pulse_waveform_upsampled_list, axis=0)

    return t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, True


def generate_t2(t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, upper_ratio=0.10):
    """
    t1と各波形のMSEを比較し，MSEが一定閾値（例: 0.01）未満の波形のみを平均してt2を生成。
    
    Parameters
    ----------
    t1 : np.ndarray
        平均波形t1
    pulse_waveform_upsampled_list : np.ndarray
        アップサンプリングされた個々の波形（2次元）
    pulse_waveform_original_list : np.ndarray
        元波形（0埋め済み）のリスト（プロット用）
    upper_ratio : float
        （現在未使用）MSE上位xx%を除外するオプションとして保持

    Returns
    -------
    t2 : np.ndarray
        MSE選別後の平均波形
    t2_plot : np.ndarray
        プロット用の平均波形（原波形ベース）
    """
    if t1 is None or pulse_waveform_upsampled_list is None:
        print("Error: t1 or pulse_waveform_upsampled_list is None.")
        return None, None
    
    mse_list = np.array([mean_squared_error(t1, wave) for wave in pulse_waveform_upsampled_list])

    # 上位10%を除外 → 今はMSEが0.01未満のみ採用
    mse_threshold = 0.01
    filtered_idx = np.where(mse_list < mse_threshold)[0]

    if len(filtered_idx) == 0:
        print("Warning: No waveforms passed MSE threshold.")
        return None, None

    # t2計算
    t2 = np.mean(pulse_waveform_upsampled_list[filtered_idx], axis=0)
    
    pulse_for_t2_plot = pulse_waveform_original_list[filtered_idx]
    t2_plot = np.mean(pulse_for_t2_plot, axis=0)
    plt.rcParams["font.size"] = 14
    
    ## 可視化用関数
    # for pulse_for_t2 in pulse_for_t2_plot:
    #     plt.plot(pulse_for_t2, color="skyblue")
                
    # plt.plot(t2_plot, color="orange", lw=3)
    # plt.xlabel("Frame")
    # plt.ylabel("Amplitude")
    # plt.title("Average Pulse (t2)")
    # plt.tick_params(labelsize=12)
    # plt.tight_layout()
    # # plt.show()
    # # plt.close()
    
    return t2