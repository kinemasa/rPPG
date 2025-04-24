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
import re

""" サードパーティライブラリのインポート """
import numpy as np
import csv
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import cv2 
import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

""" 自作ライブラリのインポート """
#sys.path.append('/Users/ik/GglDrv2/FY-2021/11_KnowledgeTransfer/EstBP-AI/Software') # 自作ライブラリを格納しているフォルダのパスを通す．
sys.path.append('../')
import basic_process.make_output_folder as mof  # 出力データを格納するフォルダを作成するためのモジュール
import basic_process.basic_loading_process as blp  # 基本的なデータ入力処理のためのモジュール


INPUT_FOLDER = 'c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-data-output-test2'
INPUT_FOLDER_ORIGINAL = 'c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-dataset-test'

OUTPUT_FOLDER = 'c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-result-test2'
#OUTPUT_FOLDER = '/Volumes/My_SSD/3_M2/修士論文/可視化/資料用/Compare_ROI'
os.makedirs(OUTPUT_FOLDER,exist_ok=True)
# 入力データとして，ファイルを用いるかフォルダを用いるかを指定するフラグ
USE_FOLDER_FOR_INPUT = True  # True: 入力データとしてフォルダを指定する | False: 入力データとしてファイルを指定する．
USE_FOLDER_FOR_INPUT_O = True

INPUT_FILES = np.array([])  # 脈波データ群のファイルパス
INPUT_FILES_ORIGINAL = np.array([])

# ファイル・フォルダで指定した入力データ群の中で，データ群の全てを使用するか，一部を使用するかを指定するフラグ
USE_ALL_FILES = True # True: データ群を全て使用する | False: データ群の中で，USE_DATA_INDXSに指定したファイルを使用する
USE_FILE_INDXS = np.array([3])  # USE_ALL_FILESがFalseの場合，入力データ群の中で使用するファイルのインデックス

#SAMPLE_RATE = 160 # 脈波のサンプルレート[fps]
SAMPLE_RATE = 30 # 脈波のサンプルレート[fps]
START_TIME = 0

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def thermo(value):
    # 0.0～1.0 の範囲の値をサーモグラフィみたいな色にする
    # 0.0                    1.0
    # 青    水    緑    黄    赤
    # 最小値以下 = 青
    # 最大値以上 = 赤
    r = 0
    g = 0
    b = 0
    tmp_val = math.cos(4 * math.pi * value)
    col_val = (int)( ( -tmp_val / 2 + 0.5 ) * 255 )
    if value >= ( 4.0 / 4.0 ):
        r = 255
        g = 0
        b = 0 # 赤
    elif value >= ( 3.0 / 4.0 ):
        r = 255
        g = col_val
        b = 0 # 黄～赤
    elif value >= ( 2.0 / 4.0 ):
        r = col_val
        g = 255
        b = 0 # 緑～黄
    elif value >= ( 1.0 / 4.0 ):
        r = 0
        g = 255
        b = col_val # 水～緑
    elif value >= ( 0.0 / 4.0 ):
        r = 0
        g = col_val
        b = 255 # 青～水
    else:
        r = 0
        g = 0
        b = 255 # 青
    return np.array([b,g,r])
def make_mask_mediapipe_polygon(img):
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 顔の外周を構成する輪郭点インデックス（MediaPipeの仕様）
                FACE_OUTLINE_IDXS = [
                    10, 338, 297, 332, 284, 251, 389, 356, 454,
                    323, 361, 288, 397, 365, 379, 378, 400, 377,
                    152, 148, 176, 149, 150, 136, 172, 58, 132,
                    93, 234, 127, 162, 21, 54, 103, 67, 109
                ]
                points = []
                for idx in FACE_OUTLINE_IDXS:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append([x, y])
                points = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, points, 255)

    return mask
def thermo_custom(value):
    # 0.0～1.0 の範囲の値をサーモグラフィみたいな色にする
    # 0.0                    1.0
    # 青    水    緑    黄    赤
    # 最小値以下 = 青
    # 最大値以上 = 赤
    r = 0
    g = 0
    b = 0
    tmp_val = math.cos(4 * math.pi * value)
    col_val = (int)( ( -tmp_val / 2 + 0.5 ) * 255 )
    if value >= ( 3.5 / 4.0 ):
        r = 255
        g = 0
        b = 0 # 赤
    elif value >= ( 3.0 / 4.0 ):
        r = 255
        g = col_val
        b = 0 # 黄～赤
    elif value >= ( 2.0 / 4.0 ):
        r = col_val
        g = 255
        b = 0 # 緑～黄
    elif value >= ( 1.0 / 4.0 ):
        r = 0
        g = 255
        b = col_val # 水～緑
    elif value >= ( 0.5 / 4.0 ):
        r = 0
        g = col_val
        b = 255 # 青～水
    else:
        r = 0
        g = 0
        b = 255 # 青
    return np.array([b,g,r])

def make_mask(img):
    #Hue:0-360, Saturation:0-100, Value:0-100
    HSV_MIN = np.array([0, 60, 20])
    HSV_MAX = np.array([70, 170, 255])
    height, width = img.shape[:2]
    hsv_im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_im, HSV_MIN, HSV_MAX)
    return mask

'''フォルダを作成する関数'''
def make_folder(output_folder):
    os.makedirs(output_folder, exist_ok = True)

""" ROIを設定し,その部分を平均化する．画像サイズは変化させない """
def average_image(img, block_size):
    # 画像のサイズを取得
    h, w = img.shape
    
    # 高さと幅がブロックサイズの整数倍になるように画像をトリミング
    h_trim, w_trim = h - (h % block_size), w - (w % block_size)
    img_trimmed = img[:h_trim, :w_trim]
    
    # 画像をブロックごとに平均化
    reshaped = img_trimmed.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)
    block_means = reshaped.mean(axis=(1, 3))
    
    # 平均化した値を元の画像サイズに復元
    img_ave = np.repeat(np.repeat(block_means, block_size, axis=0), block_size, axis=1)
    img_ave = np.array(img_ave, dtype=np.uint8)

    # サイズを元の画像と一致させる
    img_final = np.zeros((h, w), dtype=np.float32)  # 元の画像サイズの空白を作成
    img_final[:h_trim, :w_trim] = img_ave          # トリミング部分を埋める
    img_final = np.array(img_final, dtype=np.uint8)  # 型をuint8に変換

    return img_final

def pulse_to_video(pulse_filepath, img_filepath, sample_rate, start_time, block_size, output_folder):
    num = len(pulse_filepath)
    for n in range(num):
        sys.stderr.write('\r\n# File Number : %d / %d\n' %(n+1, num))
        # ファイル名の取得
        file = pulse_filepath[n]  # パスの取得
        filename = os.path.basename(file)  # ファイル名の取得
        filename = os.path.splitext(filename)[0]  # 拡張子の除去

        #元画像のファイルパスを整理
        img_folder = img_filepath[n]
        img_filelist = sorted(glob.glob(img_folder + '/*.png'), key=natural_keys)

        img_filelist = img_filelist[sample_rate*start_time:]
        
        pulse = np.load(file) # (148, 192, 24000)

        #cv2用に変換
        pulse_norm = np.array(pulse*255, dtype=np.uint8)

        height, width, frame_num = pulse.shape
        size = (width, height)
        mov = np.zeros([pulse.shape[0], pulse.shape[1], 3, frame_num], dtype=np.uint8)
        for m in range(frame_num):
        #for m in range(160*90,160*110):
            if m % (int(frame_num/5)) == 0:
                sys.stderr.write('\r\n# Segment Number : %d / %d' % (m, frame_num))
            
            #元の画像の読み込みとリサイズ
            original_img = cv2.imread(img_filelist[m])
            original_img = cv2.resize(original_img, size, interpolation=cv2.INTER_LINEAR) #幅，高さ
            #マスク作成
            immask = make_mask_mediapipe_polygon(original_img)
            #マスク処理
            original_img = cv2.bitwise_and(original_img, original_img, mask=immask)

            #tmp = pulse_norm[:,:,m]
            #平均化
            tmp = average_image(pulse_norm[:,:,m], block_size=block_size)

            # カラーマップ化
            colormap = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
            #マスク処理
            colormap = cv2.bitwise_and(colormap, colormap, mask=immask)

            #画像の合成
            alpha = 0
            img = cv2.addWeighted(original_img, alpha, colormap, 1-alpha, 0)

            mov[:,:,:,m] = img
            
        roi_size = block_size*30
        output = output_folder+'/'+'['+ str(roi_size) + ']' +filename+'.mp4'
        size = (pulse.shape[1], pulse.shape[0])
        # fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        result = cv2.VideoWriter(output, fourcc, int(sample_rate/2), size)

        for i in range(frame_num):
            frame = mov[:, :, :, i]
            result.write(frame)

        result.release()

def pulse_to_video_using_original_colormap(pulse_filepath, img_filepath, sample_rate, start_time, block_size, output_folder):
    num = len(pulse_filepath)
    for n in range(num):
        sys.stderr.write('\r\n# File Number : %d / %d\n' %(n+1, num))
        # ファイル名の取得
        file = pulse_filepath[n]  # パスの取得
        filename = os.path.basename(file)  # ファイル名の取得
        filename = os.path.splitext(filename)[0]  # 拡張子の除去

        #元画像のファイルパスを整理
        img_folder = img_filepath[n]
        img_filelist = sorted(glob.glob(img_folder + '/*.bmp'), key=natural_keys)

        img_filelist = img_filelist[sample_rate*start_time:]

        pulse = np.load(file) # (148, 192, 24000)

        #cv2用に変換
        pulse_norm = np.array(pulse*255, dtype=np.uint8)

        height, width, frame_num = pulse.shape
        size = (width, height)
        mov = np.zeros([pulse.shape[0], pulse.shape[1], 3, frame_num], dtype=np.uint8)
        colormap = np.zeros([height, width, 3], dtype=np.uint8)
        for m in range(frame_num):
            if m % (int(frame_num/5)) == 0:
                sys.stderr.write('\r\n# Segment Number : %d / %d' % (m, frame_num))
            
            #元の画像の読み込みとリサイズ
            original_img = cv2.imread(img_filelist[m])
            original_img = cv2.resize(original_img, size, interpolation=cv2.INTER_AREA) #幅，高さ

            #マスク作成
            immask = make_mask(original_img)

            #マスク処理
            original_img = cv2.bitwise_and(original_img, original_img, mask=immask)

            #平均化
            tmp = average_image(pulse_norm[:,:,m], block_size=block_size)

            for h in range(height):
                for w in range(width):
                    # カラーマップ化
                    colormap[h,w,:] = thermo(tmp[h,w]/255)
                    #colormap[h,w,:] = thermo_custom(tmp[h,w]/255)
                    
            
            colormap = np.array(colormap, dtype=np.uint8)
            #マスク処理
            colormap = cv2.bitwise_and(colormap, colormap, mask=immask)

            #画像の合成
            alpha = 0
            img = cv2.addWeighted(original_img, alpha, colormap, 1-alpha, 0)

            mov[:,:,:,m] = img
        
        roi_size = block_size*30
        output = output_folder+'/'+'['+ str(roi_size) + ']Original-' +filename+'.mp4'
        size = (pulse.shape[1], pulse.shape[0])
        fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
        result = cv2.VideoWriter(output, fourcc, int(sample_rate/2), size)

        for i in range(frame_num):
            frame = mov[:, :, :, i]
            result.write(frame)

        result.release()

# 時間算出のための変数定義
t1 = time.time()

""" [1] 出力ファイルを格納するフォルダを作成 """
output_folder = mof.make_output_folder(OUTPUT_FOLDER, os.path.basename(__file__))

""" [2] 脈波データのファイルパスを整理 """
pulse_filepaths = blp.extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES, USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)
original_img_filepaths = blp.extract_filepaths_for_use(INPUT_FILES_ORIGINAL, USE_ALL_FILES, USE_FILE_INDXS, USE_FOLDER_FOR_INPUT_O, INPUT_FOLDER_ORIGINAL)

""" [3] 脈波読み込み """
block_size=1
pulse_to_video(pulse_filepaths, original_img_filepaths, SAMPLE_RATE, START_TIME, block_size, output_folder)
#pulse_to_video_using_original_colormap(pulse_filepaths, original_img_filepaths, SAMPLE_RATE, START_TIME, block_size, output_folder)

# 要した時間を標準出力
t2 = time.time()
elapsed_time = int(t2 - t1)

print(f'Time : {elapsed_time} sec')

print('\r\nThis program has been finished successfully!')


