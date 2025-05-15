"""
脈波に対する基本的な処理

20200821 Kaito Iuchi
"""


""" 標準ライブラリのインポート """
import glob
import os
import sys
import time
import numba
import math

""" サードバーティライブラリのインポート """
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import eye
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy.sparse import csc_matrix, lil_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn


def calculate_snr(pulse, sample_rate, output_folder):
    """
    脈波データからSN比を算出する．

    Parameters
    ---------------
    pulse : np.ndarray (1dim)
        脈波データ
    sample_rate : int
        脈波のサンプルレート
    output_folder : string
        出力フォルダ

    Returns
    ---------------
    snr : np.float
        SN比

    """
    
    sn = pulse.shape[0]
    psd = np.fft.fft(pulse)
    psd = np.abs(psd/sn/2)
    frq = np.fft.fftfreq(sn, d=1/sample_rate)
    
    sn_hlf = math.ceil(sn/2)
    psd_hlf = psd[:sn_hlf]
    frq_hlf = frq[:sn_hlf]
    frq_dff = frq_hlf[1] - frq_hlf[0]

    np.savetxt(output_folder + '/PSD.csv', psd_hlf)

    peak1_indx = psd_hlf.argsort()[-1]
    peak2_indx = psd_hlf.argsort()[-2]
    
    if peak2_indx < peak1_indx:
        peak1_indx = peak2_indx

    wd = int(0.3 / frq_dff)
    
    harmonic_1st = np.arange(peak1_indx - wd, peak1_indx + wd+1)
    harmonic_2nd = np.arange(peak1_indx * 2 - wd, peak1_indx * 2 + wd+1)
    harmonic_3rd = np.arange(peak1_indx * 3 - wd, peak1_indx * 3 + wd+1)
    sgnl = np.sum(psd_hlf[harmonic_1st]) * frq_dff + np.sum(psd_hlf[harmonic_2nd]) * frq_dff + np.sum(psd_hlf[harmonic_3rd]) * frq_dff
    ttl = np.sum(psd_hlf[:]) * frq_dff
    nois = ttl - sgnl
    
    snr = 10 * np.log10(sgnl / nois)

    return snr


def calculate_hrv(pulse, sample_rate):
    """
    脈波データから心拍変動(IBI，心拍数，PSD)を算出する．

    Parameters
    ---------------
    pulse : np.ndarray (1dim)
        脈波データ
    sample_rate : int
        脈波のサンプルレート

    Returns
    ---------------
    ibi : nd.ndarray (1dim [100[fps] × 合計時間])
        IBI
    pulse_rate : np.float
        心拍数
    frq : np.ndarray (1dim)
        周波数軸
    psd : np.ndarray (1dim)
        パワースペクトル密度

    """
    
    # ピーク検出
    peak1_indx, peak2_indx = detect_pulse_peak(pulse, sample_rate)
    
    # 下側ピーク数 / 心拍変動は下側ピークで算出した方が精度が良い．
    peak_num = peak2_indx.shape[0]
    ibi = np.zeros([peak_num - 1])
    flag = np.zeros([peak_num - 1])
    
    # IBI算出
    for num in range(peak_num - 1):
        ibi[num] = (peak2_indx[num + 1] - peak2_indx[num]) / sample_rate
        
    # ibiが[0.25, 1.5][sec]の範囲内に無い場合，エラーとする．/ [0.5, 1.5]
    error_indx = np.where((ibi < 0.33) | (1.5 < ibi))
    flag[error_indx] = False
    ibi[error_indx] = np.nan

    global count_flag
    if np.any(flag):
        print('[!]')
        count_flag += 1

    ibi_num = ibi.shape[0]
    # スプライン補間は，次数以上の要素数が必要
    if ibi_num > 3:
        spln_kind = 'cubic'
    elif ibi_num > 2:
        spln_kind = 'quadratic'
    elif ibi_num > 1:
        spln_kind = 'slinear'
    else:
        ibi = np.nan

    total_time = np.sum(ibi)
    if np.isnan(total_time) != True:
        # エラーが発生した箇所の補間
        ibi = btp.interpolate_nan(ibi)
        total_time = np.sum(ibi)
        # 心拍数の算出
        pulse_rate = 60 / np.mean(ibi)
        # スプライン補間 / 1fpsにリサンプリング
        # リサンプリングレート
        fs = 2
        sample_num = int(total_time * fs)
        x = np.linspace(0.0, total_time, ibi_num)
        f_spln = interp1d(x, ibi, kind=spln_kind)
        x_new = np.linspace(0.0, int(total_time), sample_num)
        ibi_spln = f_spln(x_new)

        sn = ibi_spln.shape[0]
        psd = np.fft.fft(ibi_spln)
        psd = np.abs(psd)
        frq = np.fft.fftfreq(n=sn, d=1/fs)
        sn_hlf = math.ceil(sample_num / 2)
        psd = psd[:sn_hlf]
        frq = frq[:sn_hlf]

    else:
        ibi = np.nan
        pulse_rate = np.nan
        frq = np.nan
        psd = np.nan

    return ibi, pulse_rate, frq, psd


def scndiff_pulse(one_pulse, resample_rate):
    """
    1脈動分の脈波を2皆微分する．

    Parameters
    ---------------
    pulse : np.float (1 dim)
        1脈動分の脈波
    resample_rate : int
        スプライン補間でリサンプリングする際のサンプルレート

    Returns
    ---------------
    pulse_nrml : np.float
        正規化した1脈動分の脈波
    peak_indx : int
        脈波データの上側ピークのインデックス
    flag : int
        使えるデータか使えないデータかを格納する． / True : 使用可能, False : 使用不可
    
    """
    
    one_pulse_length = one_pulse.shape[0]
    
    # 500フレームにリサンプリング
    x = np.linspace(0.0, 1.0, one_pulse_length)
    f_spln = interp1d(x, one_pulse, kind='cubic')
    x_new = np.linspace(0.0, 1.0, resample_rate)
    one_pulse = f_spln(x_new)

    # デトレンド処理    
    x = [0, resample_rate - 1]
    y = [one_pulse[0], one_pulse[-1]]
    
    res = np.polyfit(x, y, 1)
    x2 = np.arange(0, resample_rate)
    poly = np.poly1d(res)(x2)
    
    frst_drv = np.diff(one_pulse) * resample_rate
    scnd_drv = np.diff(frst_drv) * resample_rate
    
    scnd_drv = btp.polyfit_data(scnd_drv, resample_rate, 15)

    flag = True
    return frst_drv, scnd_drv, flag


def normalize_pulse(one_pulse, resample_rate, upper_pulse):
    """
    1脈動分の脈波を正規化する．

    Parameters
    ---------------
    pulse : np.float (1 dim)
        1脈動分の脈波
    resample_rate : int
        スプライン補間でリサンプリングする際のサンプルレート
    upper_pulse : boolean
        True : 上向き脈波 / False : 下向き脈波

    Returns
    ---------------
    pulse_nrml : np.float
        正規化した1脈動分の脈波
    peak_indx : int
        脈波データの上側ピークのインデックス
    flag : int
        使えるデータか使えないデータかを格納する． / True : 使用可能, False : 使用不可
    
    """
    
    one_pulse_length = one_pulse.shape[0]
    flag = True
    
    # 500フレームにリサンプリング
    x = np.linspace(0.0, 1.0, one_pulse_length)
    f_spln = interp1d(x, one_pulse, kind='cubic')
    x_new = np.linspace(0.0, 1.0, resample_rate)
    one_pulse = f_spln(x_new)

    # デトレンド処理    
    x = [0, resample_rate - 1]
    y = [one_pulse[0], one_pulse[-1]]
    
    res = np.polyfit(x, y, 1)
    x2 = np.arange(0, resample_rate)
    poly = np.poly1d(res)(x2)
    
    pulse_nrml = np.abs(one_pulse - poly)

    # マイナス値を0にする．
    pulse_nrml[pulse_nrml < 0] = 0
    
    # ピーク検出
    peak_indx = np.argmax(pulse_nrml)
    
    # ピークが時間的に遅い場合はエラーだと見なす． / 下向き脈波の場合は早すぎる場合
    if upper_pulse == True:
        if peak_indx > int(resample_rate * 0.6):
            flag = False
    else:
        if peak_indx < int(resample_rate * 0.4):
            flag = False
    
    pulse_nrml = pulse_nrml / pulse_nrml[peak_indx]   

    # 下向き脈波の場合は波形を反転する．
    if upper_pulse == False:
        pulse_nrml = -1 * pulse_nrml + pulse_nrml[peak_indx]

    return pulse_nrml, peak_indx, flag


def detrend_pulse(pulse, sample_rate):
    """
    脈波をデトレンドする．
    脈波が短すぎるとエラーを出力．(T < wdth の場合)
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_dt : np.float (1 dim)
        デトレンドされた脈波
    
    """
    @ numba.jit
    def inv_jit(A):
        return np.linalg.inv(A)

    t1 = time.time()
    
    # デトレンドによりデータ終端は歪みが生じるため，1秒だけ切り捨てる．    
    pulse_length = pulse.shape[0]
    print(pulse_length)
    pulse = np.concatenate([pulse, pulse[-2 * sample_rate:]])
    virt_length = pulse_length + 2 * sample_rate
    # デトレンド処理 / An Advanced Detrending Method With Application to HRV Analysis
    pulse_dt = np.zeros(virt_length)
    order = len(str(virt_length))
    print(order)
    lmd = sample_rate * 12 # サンプルレートによって調節するハイパーパラメータ

    wdth = sample_rate * 2 # データ終端が歪むため，データを分割してデトレンドする場合，wdth分だけ終端を余分に使用する．

    if order > 4:
        splt = int(sample_rate / 16) # 40
        T = int(virt_length/splt)
        # wdth = T
        for num in range(splt):
            print('\r\t[Detrending Pulse] : %d / %d' %(num + 1, splt), end='')
            if num < (splt - 1):
                I = np.identity(T + wdth)
                flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
                D2 = spdiags(flt.T, np.array([0, 1, 2]), T + wdth - 2, T + wdth)
                preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
                inv_tmp = inv_jit(preinv)
                tmp = (I - inv_tmp) @ pulse[num * T : (num + 1) * T + wdth]
                tmp = tmp[0 : -wdth]
                pulse_dt[num * T : (num + 1) * T] = tmp
            else:
                # I = eye(T, T)
                I = np.identity(T)
                flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
                D2 = spdiags(flt.T, np.array([0, 1, 2]), T - 2, T)
                preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
                inv_tmp = inv_jit(preinv)
                tmp = (I - inv_tmp) @ pulse[num * T: (num + 1) * T]
                pulse_dt[num * T : (num + 1) * T] = tmp

    else:
        T =len(pulse)
        pulse_dt = np.zeros(len(pulse))
        I = np.identity(T)
        flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
        D2 = spdiags(flt.T, np.array([0, 1, 2]), T - 2, T)
        preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
        inv_tmp = inv_jit(preinv)
        pulse_dt[:] = (I - inv_tmp) @ pulse

    pulse_dt = pulse_dt[0 : len(pulse)-sample_rate*2]

    t2 = time.time()
    elapsed_time = int((t2 - t1) * 10)
    print(f'\tTime : {elapsed_time * 0.1} sec')
    
    return pulse_dt


def sg_filter_pulse(pulse, sample_rate):
    """
    SGフィルタリングにより脈波をデノイジングする．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_sg : np.float (1 dim)
        デノイジングされた脈波
    
    """ 
    
    # SGフィルタリング
    pulse_sg = signal.savgol_filter(pulse, int(sample_rate / 2.0) + 1, 5)
    
    return pulse_sg


def bandpass_filter_pulse(pulse, band_width, sample_rate):
    """
    バンドパスフィルタリングにより脈波をデノイジングする．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    band_width : float (1dim / 2cmps)
        通過帯 [Hz] (e.g. [0.75, 4.0])
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_sg : np.float (1 dim)
        デノイジングされた脈波
    
    """ 
    
    # バンドパスフィルタリング
    nyq = 0.5 * sample_rate
    b, a = signal.butter(1, [band_width[0]/nyq, band_width[1]/nyq], btype='band')
    pulse_bp = signal.filtfilt(b, a, pulse)
    
    return pulse_bp


def detect_pulse_peak(pulse, sample_rate):
    """
    脈波の上側ピークと下側ピークを検出する．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    peak1_indx : int (1 dim)
        脈波の上側ピーク
    peak2_indx : int (1 dim)
        脈波の下側ピーク
    
    """ 
    
    # ピーク検出
    peak1_indx = signal.argrelmax(pulse, order=int(sample_rate / 3.0))[0]
    peak2_indx = signal.argrelmin(pulse, order=int(sample_rate / 3.0))[0]
    
    return peak1_indx, peak2_indx


def reverse_pulse(pulse):
    """
    脈波の振幅を逆転させる．
    
    Parameters
    ---------------
    pulse : ndarray (1dim)
        脈波データ

    Returns
    ---------------
    pulse : ndarray (1dim)
        振幅を逆転させた脈波データ
    
    """ 
    
    # iPPGは脈波の振幅が逆になっているからその修正
    pulse = pulse * -1
    pulse = pulse - np.min(pulse)
    
    return pulse


# def preprocess_pulse(pulse, sample_rate):
#     """
#     脈波に対して前処理を行う．
#     振幅逆転 > デトレンド > SGフィルタ > バンドパスフィルタ > ピーク検出
    
#     Parameters
#     ---------------
#     pulse_lst : list(1dim : np.ndarray (2dim, [フレーム数, 波長数]))
#         ファイルごとの脈波
#     sample_rate : int
#         脈波のサンプルレート

#     Returns
#     ---------------
#     pulse_dt : np.ndarray (2dim, [フレーム数, 波長数]))
#         デトレンド脈波
#     pulse_sg : np.ndarray (2dim, [フレーム数, 波長数]))
#         SGフィルタリング脈波
#     pulse_bp : np.ndarray (2dim, [フレーム数, 波長数]))
#         BPフィルタリング脈波
#     peak1_indx : np.ndarray (1dim, [ピーク数]))
#         上側ピーク
#     peak2_indx : np.ndarray (2dim, [ピーク数]))
#         下側ピーク
    
#     """ 

#     # print('[Preprocessing Pulse]\n')
    
#     # # 脈波に0が含まれている場合，近傍を用いて補間する．
#     # pulse[pulse==0] = np.nan
#     # pulse = btp.interpolate_nan(pulse)
    
#     # # 脈波に異常値が含まれている場合，近傍を用いて補間する．
#     # pulse, outlier_indx = btp.interpolate_outlier(pulse, True, th_constant=2.5)
    
#     # 脈波の振幅を逆転させる．
#     pulse = bpp.reverse_pulse(pulse)
    
#     # デトレンド / 脈波の終端切り捨てが発生することに注意
#     pulse_dt = bpp.detrend_pulse(pulse, sample_rate)
    
#     # SG-フィルタリング
#     pulse_sg = bpp.sg_filter_pulse(pulse_dt, sample_rate)
    
#     # バンドパスフィルタリング / [0.75, 5.0]
#     band_width = [0.75, 8.0]
#     pulse_bp = bpp.bandpass_filter_pulse(pulse_sg, band_width, sample_rate)
    
#     # ピーク検出
#     peak1_indx, peak2_indx = bpp.detect_pulse_peak(pulse_bp, sample_rate)
            
#     return pulse_dt, pulse_sg, pulse_bp, peak1_indx, peak2_indx

