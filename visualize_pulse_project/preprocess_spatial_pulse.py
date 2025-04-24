# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:04:38 2024

@author: imai
"""

""" 標準ライブラリのインポート """
import glob
import os
import sys
import functools
import time
import copy

""" サードパーティライブラリのインポート """
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import signal
import pandas as pd 
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags
import scipy.sparse.linalg as spla

""" 自作ライブラリのインポート """
#sys.path.append('/Users/ik/GglDrv2/FY-2021/11_KnowledgeTransfer/EstBP-AI/Software') # 自作ライブラリを格納しているフォルダのパスを通す．
# sys.path.append('../')
import basic_process.make_output_folder as mof  # 出力データを格納するフォルダを作成するためのモジュール
import basic_process.basic_loading_process as blp  # 基本的なデータ入力処理のためのモジュール

INPUT_FOLDER = 'c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-data-output-test\\'

OUTPUT_FOLDER = 'c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-data-output-test2'
os.makedirs(OUTPUT_FOLDER,exist_ok=True)
INPUT_FILES = np.array([])  # 脈波データ群のファイルパス

# 入力データとして，ファイルを用いるかフォルダを用いるかを指定するフラグ
USE_FOLDER_FOR_INPUT = True  # True: 入力データとしてフォルダを指定する | False: 入力データとしてファイルを指定する．

# ファイル・フォルダで指定した入力データ群の中で，データ群の全てを使用するか，一部を使用するかを指定するフラグ
USE_ALL_FILES = True # True: データ群を全て使用する | False: データ群の中で，USE_DATA_INDXSに指定したファイルを使用する
USE_FILE_INDXS = np.array([])  # USE_ALL_FILESがFalseの場合，入力データ群の中で使用するファイルのインデックス

#SAMPLE_RATE = 160 # 脈波のサンプルレート[fps]
SAMPLE_RATE = 30

BAND_WIDTH = [0.75, 4.0] # バンドパスの通過周波数帯域

EXT_RANGE = [0,3]

#paramSG = [30, 5]
paramSG = [15, 5]

def detrend_pulse(pulse, sample_rate, crop_range):
    #pulse : (フレーム数,)
    exp_lng = crop_range[1]-crop_range[0]
    #トレンド推定　多項式フィッティング
    t = np.arange(0, exp_lng, 1/sample_rate)
    
    if len(pulse) != len(t):
        print(f"[ERROR] pulse and t length mismatch: len(pulse)={len(pulse)}, len(t)={len(t)}")
        min_len = min(len(pulse), len(t))
        pulse = pulse[:min_len]
        t = t[:min_len]

    #coefficients = np.polyfit(t, pulse, deg=5)
    coefficients = np.polyfit(t, pulse, deg=15)
    trend = np.polyval(coefficients, t)

    #デトレンド後波形
    pulse_dt = pulse - trend
    
    return pulse_dt

#Savitzky-Golyによるノイズ除去
def SGs(y,dn,poly):
    # y as np.array, dn as int, poly as int
    n = len(y) // dn
    if n % 2 == 0:
        N = n+1
    elif n % 2 == 1:
        N = n
    else:
        print("window length can't set as odd")
    SGsmoothed = signal.savgol_filter(y, window_length=N, polyorder=poly)
    return SGsmoothed

def sg_filter_pulse(pulse, paramSG):
    pulse_sg = SGs(pulse, paramSG[0], paramSG[1])
    
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
    # カットオフ周波数を正規化
    low = band_width[0] / nyq
    high = band_width[1] / nyq
    # 正規化周波数の範囲が0 < Wn < 1になるようにチェック
    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError("サンプルレートに対して適切なバンド幅を設定してください。")
    # バンドパスフィルタを設計
    b, a = signal.butter(1, [low, high], btype='band')
    pulse_bp = signal.filtfilt(b, a, pulse)
    
    return pulse_bp

def process_pulsewave(pulse2d_lst, sample_rate, bandwidth, paramSG, crop_range, use_filepaths, output_folder):

    """ [1] データ保存用リスト・配列の宣言 """
    pulse_num = len(pulse2d_lst)  # 脈波データ数

    """ [2] 脈波データの数だけ畳み込み処理を行う． """
    for num in range(pulse_num):
        
        sys.stderr.write('\r\n# File Number : %d / %d\n' %(num+1, pulse_num))
        #print('\r\n===== File Number : %d / %d =====' %(num+1, pulse_num))  # 何番目のデータを処理しているかを標準出力

        # ファイル名の取得
        file = use_filepaths[num]  # パスの取得
        filename = os.path.basename(file)  # ファイル名の取得
        filename = os.path.splitext(filename)[0]  # 拡張子の除去     
        print(filename)
        
        """ [3] 脈波データ群が格納されたリストから1つの脈波データを取り出す． """
        pulse2d = pulse2d_lst[num]

        p2d_height = pulse2d.shape[1]
        p2d_width = pulse2d.shape[2]

        processed_pulse = np.zeros([p2d_height, p2d_width, (crop_range[1]-crop_range[0])*sample_rate])
        
        min_max_scaler = MinMaxScaler(feature_range=(0,1))

        count = 0
        for n1 in range(p2d_height):
            for n2 in range(p2d_width):
                count += 1
                if count % (int(p2d_height * p2d_width/7)) == 0:
                    sys.stderr.write('\r\n# Segment Number : %d / %d' % (count, p2d_height * p2d_width))
                    #print('\r\n# Segment Number : %d / %d' % (count, p2d_height * p2d_width))  # 何番目のデータを処理しているかを標準出力

                pulse = pulse2d[sample_rate*crop_range[0]:sample_rate*crop_range[1], n1, n2]
                
                # SG-フィルタリング
                pulse_sg = sg_filter_pulse(pulse, paramSG)

                # デトレンド
                pulse_dt = detrend_pulse(pulse_sg, sample_rate, crop_range)
                
                # バンドパスフィルタリング / [0.75, 8.0]
                pulse_bp = bandpass_filter_pulse(pulse_dt, bandwidth, sample_rate)

                # 脈波の振幅を逆転させる．
                #pulse_bp = bpp.reverse_pulse(pulse_bp)
                
                #正規化 (0~1)
                pulse_nm = min_max_scaler.fit_transform(pulse_bp.reshape(-1,1))
                
                #標準化
                #pulse_nm = std_scaler.fit_transform(pulse_bp.reshape(-1,1))
                
                processed_pulse[n1, n2, :] = pulse_nm[:,0]

        np.save(output_folder + '/'+ filename + '.npy', processed_pulse)

    return None


# 時間算出のための変数定義
t1 = time.time()

""" [1] 出力ファイルを格納するフォルダを作成 """
output_folder = mof.make_output_folder(OUTPUT_FOLDER, os.path.basename(__file__))

# inputとしてのファイルパスを整理
""" [2] 脈波データのファイルパスを整理 """
pulse_filepaths = blp.extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES, USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)

""" [3] 脈波データの整理 """
data_num = len(pulse_filepaths)
pulse2d_lst = list()
for n1 in range(data_num):
    pulse2d = np.load(pulse_filepaths[n1])
    pulse2d_lst.append(pulse2d)


""" [4] 脈波特徴量の抽出 """
process_pulsewave(pulse2d_lst, SAMPLE_RATE, BAND_WIDTH, paramSG, EXT_RANGE, pulse_filepaths, output_folder)

# 要した時間を標準出力
t2 = time.time()
elapsed_time = int(t2 - t1)

print(f'Time : {elapsed_time} sec')

print('\r\nThis program has been finished successfully!')