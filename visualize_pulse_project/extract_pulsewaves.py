"""
空間脈波を抽出する．

20190716 Kaito Iuchi
"""

""" 標準ライブラリのインポート """
import os
import sys
import gc
import time
import re
import cv2
import glob

""" サードパーティライブラリのインポート """
import numpy as np
from PIL import Image

""" 自作ライブラリのインポート """
# 自作ライブラリを格納しているフォルダのパスを通す．
sys.path.append('../../')
import basic_process.make_output_folder as mof  # 出力データを格納するフォルダを作成するためのモジュール
import basic_process.basic_loading_process as blp  # 基本的なデータ入力処理のためのモジュール
import basic_process.funcSkinSeparation as fss  # 色素成分分離処理のためのモジュール

""" 定数設定 """
USE_FOLDER_FOR_INPUT = True # True: 入力データとしてフォルダを指定する | False: 入力データとしてファイルを指定する．
# USE_FOLDER_FOR_INPUTがTrueの場合，これら変数で指定するフォルダ下のファイルを全て入力する．

INPUT_FILES = np.array([])

# USE_FOLDER_FOR_INPUTがTrueの場合，これら変数で指定するフォルダ下のファイルを全て入力する．
INPUT_FOLDER = "c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-dataset-test"

OUTPUT_FOLDER = 'c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-data-output-test'
os.makedirs(OUTPUT_FOLDER,exist_ok=True)


# ファイル・フォルダで指定した入力データ群の中で，データ群の全てを使用するか，一部を使用するかを指定するフラグ
USE_ALL_FILES = True # True: データ群を全て使用する | False: データ群の中で，USE_DATA_INDXSに指定したファイルを使用する．
# USE_ALL_FILESがFalseの場合，入力データ群の中で使用するファイルのインデックス
USE_FILE_INDXS = np.array([5])

SEG = 3 #　平均化フィルタのサイズ
STRIDE = 1 # スライドサイズ

# 平均値フィルタを適用する関数
def mean_filter(img, kernel_size, stride):
    """
    平均値フィルタを画像に適用する。
    フィルタサイズはkernel_size x kernel_size、スライド幅はstride。

    Parameters:
        hem_img (np.ndarray): フィルタを適用する画像 (height, width, channels)
        kernel_size (int): フィルタサイズ（奇数）
        stride (int): スライド幅

    Returns:
        np.ndarray: フィルタ適用後の画像
    """
    # 画像の周囲をpadding (最外のピクセルで拡張)
    pad_size = kernel_size // 2
    padded_img = cv2.copyMakeBorder(
        img, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_REPLICATE
    )

    # 出力画像のサイズを計算
    out_height = (padded_img.shape[0] - kernel_size) // stride + 1
    out_width = (padded_img.shape[1] - kernel_size) // stride + 1
    out_img = np.zeros((out_height, out_width, img.shape[2]), dtype=img.dtype)

    # 各チャンネルごとにフィルタを適用
    for c in range(img.shape[2]):
        channel = padded_img[:, :, c]
        filtered_channel = np.zeros((out_height, out_width), dtype=channel.dtype)
        for i in range(0, out_height * stride, stride):
            for j in range(0, out_width * stride, stride):
                # カーネル部分を取得して平均値を計算
                kernel = channel[i:i + kernel_size, j:j + kernel_size]
                filtered_channel[i // stride, j // stride] = np.mean(kernel)
        out_img[:, :, c] = filtered_channel

    return out_img

def calc_img_size(img, kernel_size, stride):
    pad_size = kernel_size // 2
    padded_img = cv2.copyMakeBorder(
        img, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_REPLICATE
    )

    # 出力画像のサイズを計算
    out_height = (padded_img.shape[0] - kernel_size) // stride + 1
    out_width = (padded_img.shape[1] - kernel_size) // stride + 1
    return out_height, out_width


def extract_ippg_2d(filepath, output_folder, size_kernel, size_stride):
    """ [1] データ数を確認 """
    data_num = len(filepath)

    """ [2] データ数分だけROIの設定を行う．"""
    for n in range(data_num):

        t1 = time.time()  # 時間確認のための変数用意
        # 処理対象のデータの何番目かを標準出力
        sys.stderr.write('\r\n===== File Number : %d / %d =====\n' %(n + 1, data_num))

        """ [3] ファイル名取得・画像群ごとに処理 """
        # ファイル名の取得
        file = filepath[n]  # パスの取得
        filename = os.path.basename(file)  # ファイル名の取得
        filename = os.path.splitext(filename)[0]  # 拡張子の除去
        # bmpファイルリスト
        imgs = sorted(glob.glob(file + '/*.png'))

        img_tmp = cv2.imread(imgs[0])
        height, width = calc_img_size(img_tmp, kernel_size=size_kernel, stride=size_stride)
        print('Heihgt : ', height)
        print('Width : ', width)
        
        # フレーム数　取得
        frame_num = len(imgs)

        pulse_2d = np.zeros([frame_num, height, width])
        """ [4] 画像群の各フレームのROIを平均化 """
        for a in range(frame_num):
            # フレームを読み込む
            data_tmp = cv2.imread(imgs[a])

            # 色素成分分離
            hem_img = fss.skinSeparation(data_tmp) #hem_img : (height, width, 3)

            # 平均化
            ave_img = mean_filter(hem_img, kernel_size=size_kernel, stride=size_stride)

            # RGBチャンネルの平均値を計算
            pulse_2d[a, :, :] = np.mean(ave_img, axis=2)

            # 何フレーム目かを標準出力
            t2 = time.time()
            elapsed_time = t2 - t1
            print('\r\t[Extracting Pulse] : %d / %d : %d sec' % (a + 1, frame_num, elapsed_time), end='')

        # 保存
        np.save(output_folder + '/Pulse2D-' + filename + '.npy', pulse_2d)

        del pulse_2d
        gc.collect()

        # 要した時間を標準出力
        t2 = time.time()
        elapsed_time = int(t2 - t1)
        print(f'\tTime : {elapsed_time} sec')

    return None


if __name__ == '__main__':
    """
    [1] 出力ファイルを格納するフォルダを作成
    [2] 動画データのファイルパスを整理
    [3] ROI設定・脈波抽出を行う．
    """

    """ [1] 出力ファイルを格納するフォルダを作成 """
    output_folder = mof.make_output_folder(OUTPUT_FOLDER,  os.path.basename(__file__))

    """ [2] 動画データのファイルパスを整理 """
    filepaths = blp.extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES, USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)

    """ [3] ROI設定・脈波抽出を行う． """
    extract_ippg_2d(filepaths, output_folder, size_kernel=SEG, size_stride=STRIDE)

    print('This program has been finished successfully!')
    