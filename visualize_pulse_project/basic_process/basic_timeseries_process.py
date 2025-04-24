"""
時系列データに対する基本的な処理

20200821 Kaito Iuchi
"""


""" 標準ライブラリのインポート """
import glob
import os
import sys
import re

""" サードパーティライブラリのインポート """
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


def resample_timestamp(time_stmp, data, resample_rate, data_num=0, kind='cubic', time_range=0, start_time=0):
    """
    タイムスタンプデータを等間隔でリサンプリングする．

    Parameters
    ---------------
    time_stmp : np.ndarray (1dim [データ数])
        タイムスタンプ
    data : np.ndarray (1dim [データ数])
        時系列データ
    sample_rate : int
        リサンプルレート
    data_num : int
        データの個数

    Returns
    ---------------
    data_new : nd.ndarray (1dim [フレーム数])
        リサンプリングしたデータ

    """

    time_stmp_isstr = type(time_stmp[0]) == str 
    
    global test1
    test1 = time_stmp
    
    # タイムスタンプが数値の場合
    if time_stmp_isstr == False:

        hoge, uni_indx = np.unique(time_stmp, return_index=True)
        time_stmp = time_stmp[uni_indx]
        data = data[uni_indx]

        start_time = time_stmp[0]
        end_time = time_stmp[-1]
        time_lng = end_time - start_time
        if data_num == 0:
            x_new = np.linspace(start_time, end_time, int(resample_rate * time_lng))
        else:
            x_new = np.linspace(start_time, end_time, data_num)

        spln = interp1d(time_stmp, data, kind=kind)
        data_new = spln(x_new)
    
    # タイムスタンプが文字列(h:m:s.s)の場合
    else:  
        data_num = time_stmp.shape[0]
        time_stmp2 = np.zeros([0])
        for num in range(data_num):
            time_tmp = re.findall(r'\d+', time_stmp[num])
            time_tmp = np.array(time_tmp, dtype=np.float)
            
            time = time_tmp[0] * 60 * 60 + time_tmp[1] * 60 + time_tmp[2] + time_tmp[3] * 0.001 # h:m:s.s
            
            time_stmp2 = np.concatenate([time_stmp2, [time]])
                
        if isinstance(time_range, list):
            time_strt = int(time_range[0])
            time_lng = int(time_range[1])
        else:
            time_strt = int(time_stmp2[0])
            time_lng = int(time_stmp2[-1])

        hoge, uni_indx = np.unique(time_stmp2, return_index=True)
        time_stmp2 = time_stmp2[uni_indx]
        data = data[uni_indx]
    
        x_new = np.linspace(time_strt + start_time, time_lng, int(resample_rate * (time_lng - time_strt - start_time)))
        spln = interp1d(time_stmp2, data, kind=kind, fill_value='extrapolate')
        data_new = spln(x_new)
    
    return data_new
    

def interpolate_outlier(data, flag_intplt, th_constant=3):
    """
    時系列データの異常値の検出と置換を行う．

    Parameters
    ---------------
    data : np.ndarray (1dim / [データ長])
        時系列データ
    flag_intplt : bool
        補間するかnp.nanを置換するかのフラグ / True: 補間, False: np.nan
    th_constant : np.int
        正常値と異常値の閾値を決めるための定数

    Returns
    ---------------
    data_new : np.ndarray (2dim / [データ長])
        異常値置換後の時系列データ
    indx : np.ndarray (1dim)
        異常値のインデックス
    
    """
    
    # 第1四分位数の算出
    q1 = np.percentile(data, 25)
    
    # 第3四分位数の算出
    q3 = np.percentile(data, 75)
    
    # InterQuartile Range
    iqr = q3 - q1
    
    # 異常値判定のための閾値設定
    th_lwr = q1 - iqr * th_constant
    th_upr = q3 + iqr * th_constant
    
    indx = (data < th_lwr) | (th_upr < data)
    data[indx] = np.nan
    
    if flag_intplt:
        data_new = interpolate_nan(data)
    else:
        data_new = data
    
    if np.sum(indx) > 0:
        print(' [i]%d' %(np.sum(indx)), end='')
    
    return data_new, indx
    

def polyfit_data(data, sample_rate, deg):
    """
    時系列データを多項式近似する．

    Parameters
    ---------------
    data : np.ndarray (2dim / [データ数, データ長])
        時系列データ
    sample_rate : int
        時系列データのサンプルレート

    Returns
    ---------------
    data_poly : np.ndarray (2dim / [データ数, データ長])

    """
    
    if data.ndim == 1:
        
        data_lng = data.shape[0]
        x = np.linspace(0, data_lng, data_lng)
        y = data
        notnan_indx = np.isfinite(y)
        res = np.polyfit(x[notnan_indx], y[notnan_indx], deg)
        data_poly = np.poly1d(res)(x)
    
    else:
    
        data_num = data.shape[0]
        data_lng = data.shape[1]
        
        data_poly = np.zeros([data_num, data_lng])
        x = np.linspace(0, data_lng, data_lng)
        for num in range(data_num):
            y = data[num, :]
            notnan_indx = np.isfinite(y)

            res = np.polyfit(x[notnan_indx], y[notnan_indx], deg)
            data_poly[num, :] = np.poly1d(res)(x)
    
    return data_poly


def obtain_envelope(data, sample_rate, order_dntr=2.5):
    """
    時系列データから包絡線を取得する．

    Parameters
    ---------------
    data : np.ndarray (1dim)
        時系列データ
    sample_rate : int
        時系列データのサンプルレート

    Returns
    ---------------
    envlp_upr : nd.ndarray (1dim)
        時系列データの上側包絡線
    envlp_lwr : nd.ndarray (1dim)
        時系列データの下側包絡線

    """
    
    data_lngth = data.shape[0]

    # 血圧波形は2.5が良さげ
    peak1_indx = signal.argrelmax(data, order=int(sample_rate / order_dntr))[0]
    peak2_indx = signal.argrelmin(data, order=int(sample_rate / order_dntr))[0]
    
    # 異常値を近傍で補間
    if peak1_indx.size > 1:
        peak1, outlier_indx = interpolate_outlier(data[peak1_indx], True, th_constant=100)
    else:
        peak1 = np.array([np.nan])
    if peak2_indx.size > 1:
        peak2, outlier_indx = interpolate_outlier(data[peak2_indx], True, th_constant=100)
    else:
        peak2 = np.array([np.nan])

    peak1_num = peak1.shape[0]
    # スプライン補間は，次数以上の要素数が必要
    if peak1_num > 3:
        spln_kind1 = 'cubic'
    elif peak1_num > 2:
        spln_kind1 = 'quadratic'
    elif peak1_num > 1:
        spln_kind1 = 'slinear'
    else:
        spln_kind1 = 0

    peak2_num = peak2.shape[0]
    # スプライン補間は，時数以上の要素数が必要
    if peak2_num > 3:
        spln_kind2 = 'cubic'
    elif peak2_num > 2:
        spln_kind2 = 'quadratic'
    elif peak2_num > 1:
        spln_kind2 = 'slinear'
    else:
        spln_kind2 = 0


    if spln_kind2 != 0:
        spln_upr = interp1d(peak1_indx, peak1, kind=spln_kind1, fill_value=np.nan, bounds_error=False)
        x_new_upr = np.arange(data_lngth)
        envlp_upr = spln_upr(x_new_upr)
        envlp_upr = interpolate_nan(envlp_upr)
    else:
        envlp_upr = np.nan


    if spln_kind2 != 0:
        spln_lwr = interp1d(peak2_indx, peak2, kind=spln_kind2, fill_value=np.nan, bounds_error=False)
        x_new_lwr = np.arange(data_lngth)
        envlp_lwr = spln_lwr(x_new_lwr)
        envlp_lwr = interpolate_nan(envlp_lwr)
    else:
        envlp_lwr = np.nan
    
    return envlp_upr, envlp_lwr


def get_nearest_value(array, query):
    """
    配列から検索値に最も近い値を検索し，そのインデックスを返却する．

    Parameters
    ---------------
    array : 
        検索対象配列
    query : float
        検索値

    Returns
    ---------------
    indx : int
        検索値に最も近い値が格納されているインデックス

    """
    
    indx = np.abs(np.asarray(array) - query).argmin()
    value = array[indx]
    
    return indx


def interpolate_nan(data):
    """
    nanを近傍データで補間 (1次元 or 2次元)

    Parameters
    ---------------
    data : 1D [データ長] or 2D [データ数, データ長] / データ長軸で補間
        補間対象データ

    Returns
    ---------------
    data_inpl : 1D or 2D
        nanを補間したデータ

    """
    
    x = np.arange(data.shape[0])
    nan_indx = np.isnan(data)
    not_nan_indx = np.isfinite(data)
    
    # 始端，終端に1つもしくは連続してnanがあれば，左右近傍ではなく，片側近傍で補間
    count_nan_lft = 0
    count = 0
    while True:
        if not_nan_indx[count] == False:
            count_nan_lft += 1
            not_nan_indx[count] = True
        else:
            break
        count += 1
        
    count_nan_rgt = 0
    count = 1
    while True:
        if not_nan_indx[-count] == False:
            count_nan_rgt += 1
            not_nan_indx[-count] = True
        else:
            break
        count += 1
    
    if count_nan_lft > 0:
        data[:count_nan_lft] = np.nanmean(data[count_nan_lft+1:count_nan_lft+5])
    if count_nan_rgt > 0:
        data[-count_nan_rgt:] = np.nanmean(data[-count_nan_rgt-5:-count_nan_rgt-1])
    
    func_inpl = interp1d(x[not_nan_indx], data[not_nan_indx])
    data[nan_indx] = func_inpl(x[nan_indx])
        
    return data


def convolve_function_to_data(data, function, window_size, shift_size, sample_rate, features_num):
    """
    時系列データに対して，特徴量抽出関数を畳み込む．
    
    Parameters
    ---------------
    data : np.float (1 dim)
        時系列データ
    function : object
        畳み込む関数
    function_arg : list
        畳み込む関数の引数

    Returns
    ---------------
    features : np.float
        抽出した特徴量

    """
    
    data_length = data.shape[0]

    # 窓サイズ分だけ使用不可
    used_length = data_length - window_size * sample_rate
    used_time = used_length / sample_rate
    
    """ 畳み込み処理 """
    # 脈波をWINDOW_SIZE分だけ切り出し，1フレームずつずらしながら関数を畳み込み
    time = 0
    features = np.empty([0, features_num])
    while time < used_time:
        frame_now = int(time * sample_rate)
        
        features_tmp = function(data[frame_now : frame_now + int(sample_rate * window_size)])
        
        if isinstance(features_tmp, np.ndarray):
            features_tmp = features_tmp.flatten()

        if np.all(np.isnan(features_tmp)):
            features_tmp = np.full([features_num], np.nan)

        features = np.concatenate([features, [features_tmp]], axis=0)
        time += shift_size
        
        print('\r\t\t\tConvolving WindowFunction : %d / %d' %(time, used_time), end='')

    features = features.T

    print('\r')

    return features
    

if __name__ == '__main__':
    """ テスト用 """
    
    output_folder = mof.make_output_folder(OUTPUT_FOLDER, os.path.basename(__file__))
    
    # inputとしてのファイルパスを整理
    use_filepaths = blp.extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES,
                                                  USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)

    print('\r\n\nThis program has been finished successfully!')
    
    