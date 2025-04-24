"""
avi動画のROI内の平均画素値の時系列変化を出力

20190516 Kaito Iuchi
"""


import sys

import numpy as np

sys.path.append('C:\\Users\\iuchi\\GoogleDrive\\M2\\00_Research\\01_Software')
from versatile_tools.make_date_folder_ver2 import make_date_folder
from versatile_tools.extract_roi import *


""" 定数設定 """
INPUT_FILE = np.array(['G:\\Experimental_Data\\kazuki_20190626160207\\video\\kazuki_20190626160207.avi',
                       'G:\\Experimental_Data\\hiroto_20190626162310\\video\\hiroto_20190626162310.avi'])


def video_to_pxlval_tmsr(input_files):
    """
    avi動画のROI内の平均画素値の時系列を算出

    Parameters
    ---------------
    input_files : np.ndarray (1dim / [ファイル数])
        入力するavi動画のフィイルパス

    Returns
    ---------------
    pxlval_tmsrs : list (3dim / (1dim / ファイル数分のndarray (2dim / [バンド数, フレーム数]))
        ROI内の平均画素値の時系列
    
    """
    
    file_num = input_files.shape[0]
    
    pxlval_tmsrs = list([])
    video = list([])
    roi = list([])
    for num in range(file_num):
        video_tmp, roi_tmp = set_roi(input_files[num])
        video.append(video_tmp)
        roi.append(roi_tmp)
        
    for num in range(file_num):
        
        print(f'\nFileNumber : {num}')
        
        pxlval_tmsr = extract_roi(video[num], roi[num])
        pxlval_tmsr = pxlval_tmsr.mean(axis=(1, 2))
        
        pxlval_tmsrs.append(pxlval_tmsr)

    return pxlval_tmsrs


if __name__ == '__main__':
    """ テスト用 """
    
    pxlval_tmsr = video_to_pxlval_tmsr(INPUT_FILE)
    
