"""
タイムスタンプを単一数値に換算する．

20190716 Kaito Iuchi
"""

import os
import sys
import gc
import time
import re
import glob

import numpy as np

""" 自作ライブラリのインポート """
sys.path.append('/Users/ik/GglDrv2/FY-2021/02_Collaboration/DaikinIndustry/03_Presentation/202108_forDelivery/skinvideo_to_bloodpressure')  # 自作ライブラリを格納しているフォルダのパスを通す．
import basic_process.extract_roi as er
import basic_process.funcSkinSeparation as fss
import basic_process.make_output_folder as mof  # 出力データを格納するフォルダを作成するためのモジュール
import basic_process.basic_loading_process as blp  # 基本的なデータ入力処理のためのモジュール
import basic_process.basic_pulse_process as bpp  # 基本的な脈波処理のためのモジュール
import basic_process.basic_timeseries_process as btp  # 基本的な時系列処理のためのモジュール
import basic_process.basic_ndlst_process as bnp  # ndarrayとリストの相互変換のためのモジュール
import basic_process.visualize_data as vd  # 基本的なデータ可視化処理のためのモジュール


""" Constant Setting """
OUTPUT_FOLDER = '/Users/ik/GglDrv/M2/00_Research/02_Data/00_Temporary'
# 脈波ファイルと血圧ファイル / それぞれ対応関係にある．順序に留意．
INPUT_FILES = np.array([])
USE_ALL_FILES = True # True: INPUT_FILESのファイルを全て使用する / False: USE_FILE_INDXSに指定したファイルを使用する．
USE_FILE_INDXS = [0,1] # USE_ALL_FILESがTrueの場合，INPUT_FILESの中で使用するファイルのインデックス
USE_FOLDER_FOR_INPUT = False # True: Inputとしてフォルダを指定する / False: Inputとしてファイルを指定する．
INPUT_FOLDER = '/Users/ik/Desktop/20201118/test_tr/Cam 1' # USE_FOLDER_FOR_INPUTがTrueの場合，この変数で指定するフォルダ下のファイルを全て入力する．



def calculate_timestmp(filepaths, output_folder):
    """
    タイムスタンプを計算する．
    
    Parameters
    ---------------
    filepath : string
        RGB動画のファイルパス or RGB画像群のフォルダパス
    output_folder : string
        出力フォルダ
    vd_or_imgs : bool
        動画を入力するか，画像群を入力するかのフラグ / True: 動画を入力, False: 画像群を入力
    use_roi : bool
        ROIを設定するか，全領域を使用するかのフラグ
    rgbh : np.ndarray
        抽出するデータの選択フラグ / [Red, Green, Blue, Hem]

    Returns
    ---------------
    Nothing

    """
 
    data_num = len(filepaths)

    for num in range(data_num):
        folder = filepaths[num]
        files = sorted(glob.glob(folder + '/*.bmp'))
        frame_num = len(files)
        # タイムスタンプ処理
        tstmp = np.zeros([frame_num])
        for a in range(frame_num):
            path = files[a]
            file = os.path.basename(path)
            file = os.path.splitext(file)[0]
            ts_tmp = re.findall(r'\d\d\.\d\d\d|\d\d', file)
            ts_tmp = ts_tmp[4:]
            ts_tmp = [float(i) for i in ts_tmp]
            ts_tmp = ts_tmp[0]*60*60 + ts_tmp[1]*60 + ts_tmp[2]
            tstmp[a] = ts_tmp
            
        filename = os.path.basename(folder)
        filename = os.path.splitext(filename)[0]
        np.savetxt(output_folder + '//TimeStmp-' + filename + '.csv', tstmp, delimiter=',')        

    return None


if __name__ == '__main__':
    
    output_folder = mof.make_output_folder(OUTPUT_FOLDER,  os.path.basename(__file__))
    
    filepaths = blp.extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES, USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)
    calculate_timestmp(filepaths, output_folder)

    print('This program has been finished successfully!')
