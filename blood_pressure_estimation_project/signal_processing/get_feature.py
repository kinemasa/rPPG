import numpy as np
from scipy import signal
from blood_pressure_estimation_project.signal_processing.signal import bandpass_filter_pulse

def get_nearest_value(array, query):
    index = np.abs(np.asarray(array) - query).argmin()

    return index

def calc_contour_features(pulse_waveform, sampling_rate):
    """
    1波形分の脈波から概形特徴量を抽出する．

    Parameters
    ---------------
    pulse_waveform : np.ndarray
        1波形分の脈波
    sampling_rate : int
        関数に与える脈波のサンプリングレート

    Returns
    ---------------
    features_cn : np.ndarray
        概形特徴量
    """

    # 特徴量抽出
    # 0: t1 (Rising Time)
    # 1: t2 (sample rate - Rising time)
    # 2: t1 - t2
    # 3: t1 / t2
    # 4: s1 (Systolic Area)
    # 5: s2 (Diastolic Area)
    # 6: s1 + s2 (Total Area)
    # 7: s1 / (s1 + s2)
    # 8: s2 / (s1 + s2)
    # 9: s1 / s2 (reflection index)
    # 10: p1 (slope systolic)
    # 11: p2 (slope diastolic)
    # 12: Pulse Width 10%
    # 13: Pulse Width 20%
    # 14: Pulse Width 25%
    # 15: Pulse Width 30%
    # 16: Pulse Width 40%
    # 17: Pulse Width 50%
    # 18: Pulse Width 60%
    # 19: Pulse Width 70%
    # 20: Pulse Width 75%
    # 21: Pulse Width 80%
    # 22: Pulse Width 90%
    # 23: sys width 25%
    # 24: sys width 50%
    # 25: sys width 75%
    # 26: dia width 25%
    # 27: dia width 50%
    # 28: dia width 75%

    # 特徴量の算出
    peak_index = np.argmax(pulse_waveform)

    rising_time = peak_index
    t2 = sampling_rate - rising_time
    t1t2_sub = rising_time - t2
    t1t2_div = rising_time / t2
    systolic_area = np.sum(pulse_waveform[: peak_index + 1])
    diastolic_area = np.sum(pulse_waveform[peak_index:])
    total_area = systolic_area + diastolic_area
    s1s1s2 = systolic_area / total_area
    s2s1s2 = diastolic_area / total_area
    reflection_index = systolic_area / diastolic_area
    slope_sis = (pulse_waveform[peak_index] - pulse_waveform[0]) / peak_index
    slope_dia = (pulse_waveform[len(pulse_waveform) - 1] - pulse_waveform[peak_index]) / (len(pulse_waveform) - peak_index)
    sys_10p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.1)
    dia_10p = get_nearest_value(pulse_waveform[peak_index:], 0.1) + peak_index
    width_10p = (dia_10p - sys_10p)
    sys_20p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.2)
    dia_20p = get_nearest_value(pulse_waveform[peak_index:], 0.2) + peak_index
    width_20p = (dia_20p - sys_20p)
    sys_30p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.3)
    dia_30p = get_nearest_value(pulse_waveform[peak_index:], 0.3) + peak_index
    width_30p = (dia_30p - sys_30p)
    sys_40p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.4)
    dia_40p = get_nearest_value(pulse_waveform[peak_index:], 0.4) + peak_index
    width_40p = (dia_40p - sys_40p)
    sys_50p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.5)
    dia_50p = get_nearest_value(pulse_waveform[peak_index:], 0.5) + peak_index
    width_50p = (dia_50p - sys_50p)
    sys_60p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.6)
    dia_60p = get_nearest_value(pulse_waveform[peak_index:], 0.6) + peak_index
    width_60p = (dia_60p - sys_60p)
    sys_70p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.7)
    dia_70p = get_nearest_value(pulse_waveform[peak_index:], 0.7) + peak_index
    width_70p = (dia_70p - sys_70p)
    sys_80p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.8)
    dia_80p = get_nearest_value(pulse_waveform[peak_index:], 0.8) + peak_index
    width_80p = (dia_80p - sys_80p)
    sys_90p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.9)
    dia_90p = get_nearest_value(pulse_waveform[peak_index:], 0.9) + peak_index
    width_90p = (dia_90p - sys_90p)
    sys_25p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.25)
    dia_25p = get_nearest_value(pulse_waveform[peak_index:], 0.25) + peak_index
    width_25p = (dia_25p - sys_25p)
    sys_75p = get_nearest_value(pulse_waveform[:peak_index + 1], 0.75)
    dia_75p = get_nearest_value(pulse_waveform[peak_index:], 0.75) + peak_index
    width_75p = (dia_75p - sys_75p)

    features_cn = np.array([rising_time, t2, t1t2_sub, t1t2_div, systolic_area, diastolic_area,
                            total_area, s1s1s2, s2s1s2, reflection_index, slope_sis, slope_dia,
                            width_10p, width_20p, width_25p, width_30p, width_40p, width_50p,
                            width_60p, width_70p, width_75p, width_80p, width_90p,
                            sys_25p, sys_50p, sys_75p, dia_25p, dia_50p, dia_75p])

    return features_cn


def calc_dr_features(pulse_waveform, sampling_rate):
    """
    脈波導関数から特徴量を算出する．

    [1] 脈波の1次〜4次導関数を算出
    [2] 1次〜4次導関数から，特徴量となる点を算出
    [3] 脈波導関数から特徴量を算出

    Parameters
    ---------------
    pulse_waveform : np.ndarray
        脈波1波形分
    sampling_rate : int
        関数に与える脈波のサンプリングレート

    Returns
    ---------------
    features_dr : np.ndarray
        抽出した特徴量

    """

    """ [1] 脈波の1次〜4次導関数を算出 """
    dr_1st = np.diff(pulse_waveform)
    dr_2nd = np.diff(dr_1st)
    dr_3rd = np.diff(dr_2nd)
    dr_4th = np.diff(dr_3rd)
    # 4次微分にバンドパスフィルタを適用（ノイズが多いため，通過帯域は試行錯誤の上で設定）
    dr_4th = bandpass_filter_pulse(dr_4th, [0.4, 12.0], sampling_rate)

    """ [2] 1次〜4次導関数から，特徴量となる点を算出 """
    # a, b点は2次微分の最大値と最小値
    a = np.max(dr_2nd)
    a_index = np.argmax(dr_2nd)
    b = np.min(dr_2nd)
    b_index = np.argmin(dr_2nd)

    # c, d, e点は4次導関数を用いて算出
    peak_indexes_dr_4th = signal.argrelmax(dr_4th, order=int(sampling_rate / 20.0))[0]
    valley_indexes_dr_4th = signal.argrelmin(dr_4th, order=int(sampling_rate / 20.0))[0]

    # plt.plot(dr_4th)
    # plt.show()

    try:
        c_index = valley_indexes_dr_4th[1]
    except IndexError as err:
        c_index = np.argmin(dr_4th[50: 65])

    try:
        d_index = peak_indexes_dr_4th[0]
    except IndexError as err:
        # e_indexを求めるときの範囲（[90: 100]）と被らないようにするため，[70:88]と設定
        d_index = np.argmax(dr_4th[70: 88])

    try:
        e_index = valley_indexes_dr_4th[2]
    except IndexError as err:
        e_index = np.argmin(dr_4th[90: 100])

    # d点が検出されなかった場合
    if not 80 <= d_index <= 100:
        d_index = 90

    c = dr_2nd[c_index]
    d = dr_2nd[d_index]
    e = dr_2nd[e_index]

    # 3次導関数のピーク点を検出し，r, l点を検出
    peak_indexes_dr_3rd = signal.argrelmax(dr_3rd, order=int(sampling_rate / 20.0))[0]
    
    # peak点が見つからない場合
    if len(peak_indexes_dr_3rd) < 2:
        print(f"Error: Not enough peaks in dr_3rd in folder:")
        return None
    
    r_index = peak_indexes_dr_3rd[1]

    # # 3次導関数のプロット用
    # t = np.linspace(0, len(dr_3rd), len(dr_3rd))
    # plt.plot(dr_3rd)
    # plt.scatter(t[peak_indexes_dr_3rd], dr_3rd[peak_indexes_dr_3rd], marker="o", color="orange", s=30)
    # plt.show()

    # r点が検出されなかった場合
    if not 40 <= r_index <= 60:
        r_index = 50
        # r_index = np.argmax(dr_3rd[40: 60])

    # l点
    try:
        l_index = peak_indexes_dr_3rd[2]
    except IndexError as err:
        l_index = np.argmax(dr_3rd[80: 100])

    if not 80 <= l_index <= 100:
        l_index = np.argmax(dr_3rd[80: 100])

    """ [3] 脈波導関数から特徴量を算出 """
    # 2次導関数から特徴量を算出
    b_a = b / a
    c_a = c / a
    d_a = d / a
    e_a = e / a
    ageing_index = (b - c - d - e) / a
    waveform_index1 = (d - b) / a
    waveform_index2 = (c - b) / a
    ab_ad = (a - b) / (a - d)
    area_bd = np.sum(dr_2nd[b_index: d_index+1])

    # 1次導関数から特徴量を算出
    # v点は1次微分の最大値
    v = np.max(dr_1st)

    r_2 = dr_1st[r_index]
    l_2 = dr_1st[l_index]
    c_1_v = dr_1st[c_index] / v

    # 算出した特徴量を格納
    # 29, 30, 31, 32, 33,
    # 34, 35, 36, 37, 38, 39, 40, 41, 42,
    # 43, 44, 45,
    # 46, 47, 48, 49, 50
    features_dr = np.array([a_index, b_index, c_index, d_index, e_index,
                            a, b, c, d, e, b_a, c_a, d_a, e_a,
                            ageing_index, waveform_index1, waveform_index2,
                            ab_ad, area_bd, v, r_2, c_1_v])

    return features_dr