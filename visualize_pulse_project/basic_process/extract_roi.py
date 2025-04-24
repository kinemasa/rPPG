"""
動画・画像を読み込んでROIを抽出するためのモジュール

20190515 Kaito Iuchi
"""


""" 標準ライブラリのインポート """
import numpy as np
import sys
import tkinter
from tkinter import filedialog
import cv2
import os
import glob


def set_roi(input_path, vd_or_imgs, use_roi):
    """
    動画 or 画像群を読み込んで，ROIを抽出する関数
    読み込んだ動画データor画像群のファイルパスと抽出したROIの座標を返す．

    [1] 動画データ(or画像群データ)の読み込み
    [2] ROIの設定

    ：Parameters
    ---------------
    input_path : string
        入力のファイルorフォルダパス
        ・ファイルパス : 動画ファイルパス
        ・フォルダパス : 画像群を格納するフォルダパス
    vd_or_imgs : bool
        入力が動画or画像群のフラグ
    use_all : bool
        ROIを設定せずにフレームの全領域を用いる場合のフラグ

    ：Returns
    ---------------
    video : cv2オブジェクト or np.ndarray
        読み込んだ動画データor画像群フォルダパス
    roi : np.ndarray
        抽出したROIの座標

    """
    
    # 動画の場合
    if vd_or_imgs == True:

        """ [1] 動画データ(or画像群データ)の読み込み """
        video = cv2.VideoCapture(input_path)

        # エラー処理
        if video.isOpened():
            print('The video has been opened successfully!')
        else:
            print('The video cannot have been opened!')
            sys.exit()

        """ [2] ROIの設定 """
        video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = video.read()
        # use_roiがTrueなら，cv2のモジュールでROIの設定を実施
        # use_roiがFalseなら，ROIを全画面(フレームサイズ)に設定
        if use_roi == True:
            roi = cv2.selectROI(frame)  # ROI設定
            cv2.destroyAllWindows()
        else:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # フレーム幅を読み込み
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # フレーム高さを読み込み
            roi = [0, 0, width, height]  # ROIをフレームサイズに設定

        return video, roi
    
    # 画像群の場合
    else:

        """ [1] 動画データ(画像群データ)の読み込み """
        # 拡張子が.bmpのファイルのパスを読み込む
        files = sorted(glob.glob(input_path + '/*.bmp'))

        """ [2] ROIの設定 """
        num = 0
        # use_roiがTrueなら，cv2のモジュールでROIの設定を実施
        # use_roiがFalseなら，ROIを全画面(フレームサイズ)に設定
        if use_roi == True:
            # カメラの不具合で，最初何枚かのフレームの値がユニークである場合がある．
            # そのためのエラー処理として，2つ以上の値があるフレームが読み込まれるまで繰り返す．
            while True:
                img = cv2.imread(files[num])
                if len(np.unique(img)) > 1:
                    break
                num += 1
            roi = cv2.selectROI(img)  # ROI設定
            cv2.destroyAllWindows()

        else:
            img = cv2.imread(files[num])
            height, width, channel = img.shape
            roi = [0, 0, width, height]  # ROIをフレームサイズに設定
        
        return files, roi


def extract_roi_imgs(imgs, roi):
    """
    画像群の各フレームのROIを平均化し，
    ndarray[frame, channel]に変換する関数
    ついでにchannelの順序をcv2仕様のBGRからRGBに変換する．

    [1] 画像群の各フレームのROI内を平均化
    [2] BGR > RGB

    Parameters
    ---------------
    imgs : np.ndarray
        画像群
    roi :  ndarray(int)
        関心領域の座標

    Returns
    ---------------
    video : ndarray(float)
        動画データをadarrayに変換したもの

    """

    frame_num = len(imgs)
    data = np.zeros([frame_num, 3])
    """ [1] 画像群の各フレームのROIを平均化 """
    for a in range(frame_num):
        # フレームを読み込んで，ROIでトリミング
        data_tmp = cv2.imread(imgs[a])[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]
        # 空間平均化
        data[a, :] = data_tmp.mean(axis=(0,1))
        # 何フレーム目かを標準出力
        sys.stderr.write('\r[Extracting ROI] %d / %d' % (a+1, frame_num))
        sys.stderr.flush()

    """ [2] BGR > RGB """
    data = data[:, ::-1]

    return data


def extract_roi_video(video, roi):
    """
    動画オブジェクトorの各フレームのROIを平均化し，
    ndarray[frame, channel]に変換する関数
    ついでにchannelの順序をcv2仕様のBGRからRGBに変換する．

    [1] 動画データの各フレームのROI内を平均化
    [2] BGR > RGB

    Parameters
    ---------------
    video : cv2オブジェクト
        cv2.VideoCapture()で読み込んだ動画データ
    roi :  ndarray(int)
        関心領域の座標

    Returns
    ---------------
    video : ndarray(float)
        動画データをadarrayに変換したもの

    """
    
    # roi = roi.astype(np.int)
    
    frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_num = int(frame_num)

    """ [1] 動画データの各フレームのROI内を平均化 """
    # フレームの初期化
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # 平均化したRGBデータを格納する配列
    data = np.zeros([frame_num, 3])
    # フレーム数分ループ
    for a in range(frame_num):
        # フレームを読み込んで，ROIでトリミング
        data_tmp = np.array(video.read()[1])[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]
        # 空間平均化
        data[a, :] = data_tmp.mean(axis=(0,1))
        # 何フレーム目かを標準出力
        sys.stderr.write('\r[Extracting ROI] %d / %d' % (a, frame_num))
        sys.stderr.flush()

    """ [2] BGR > RGB """
    data = data[:, ::-1]

    return data


if __name__ == '__main__':
    """
    テスト用
    """

    print('This program has been done successfully!!')
