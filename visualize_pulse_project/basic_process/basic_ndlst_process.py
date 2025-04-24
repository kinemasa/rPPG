"""
複数のndarrayのリストに対する基本的な処理

20201101 Kaito Iuchi
"""


import glob
import os
import sys

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

""" 自作ライブラリのインポート """
sys.path.append('/Users/ik/GglDrv2/FY-2021/02_Collaboration/DaikinIndustry/03_Presentation/202108_forDelivery/skinvideo_to_bloodpressure')  # 自作ライブラリを格納しているフォルダのパスを通す．
import basic_process.make_output_folder as mof  # 出力データを格納するフォルダを作成するためのモジュール
import basic_process.basic_loading_process as blp  # 基本的なデータ入力処理のためのモジュール
import basic_process.basic_pulse_process as bpp  # 基本的な脈波処理のためのモジュール
import basic_process.basic_timeseries_process as btp  # 基本的な時系列処理のためのモジュール
import basic_process.basic_ndlst_process as bnp  # ndarrayとリストの相互変換のためのモジュール
import basic_process.visualize_data as vd  # 基本的なデータ可視化処理のためのモジュール


def flatten_ndlst(lst, cmp):
    """
    ndlstを平坦化する．

    Parameters
    ---------------
    lst : リスト (1dim [ndarray数] : np.ndarray)
        複数のndarrayを格納したリスト
    cmp :　int
        データの要素数
        
    Returns
    ---------------
    nd : np.ndarray
        lstを平坦化したもの

    """
    
    data_num = len(lst)
    
    nd = np.empty([cmp, 0])
    nd_lngs = np.empty([0], dtype=np.int)
    for num in range(data_num):
        tmp = lst[num]
        nd = np.concatenate([nd, tmp], axis=1)
        lng = tmp.shape[1]
        nd_lngs = np.concatenate([nd_lngs, [lng]])
    
    return nd, nd_lngs


def nd_to_ndlst(nd, lst_lngs):
    """
    平坦化ndarrayをndlstに変換する．

    Parameters
    ---------------
    nd : np.ndarray
        ndlstを平坦化したもの
    lst_lngs :　np.ndarray, 1dim [データ数]
        平坦化前のndlstの各データの長さ
        
    Returns
    ---------------
    lst : リスト (1dim [ndarray数] : np.ndarray)
        複数のndarrayを格納したリスト

    """
    
    nd = nd.T
    data_num = len(lst_lngs)
    
    lst = list([])
    indx = 0
    for num in range(data_num):
        tmp = nd[indx : indx + lst_lngs[num]]
        lst.append(tmp.T)
        indx += lst_lngs[num]
    
    return lst


if __name__ == '__main__':

    output_folder = mof.make_output_folder(OUTPUT_FOLDER, os.path.basename(__file__))
    
    # inputとしてのファイルパスを整理
    use_filepaths = blp.extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES,
                                                  USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)

    print('\r\n\nThis program has been finished successfully!')
    
    