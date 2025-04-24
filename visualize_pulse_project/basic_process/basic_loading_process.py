"""
データ入力に関する基本的な処理

20200812 Kaito Iuchi
"""


""" 標準ライブラリのインポート """
import glob
import re

""" サードパーティライブラリのインポート """
import numpy as np
import pandas as pd


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_data_from_filepaths_verpd(filepaths, skiprows=0, usecols=None):
    """
    ファイルパス群からデータ群を入力として読み込む．
    数値データと文字列データが混在している場合．

    [1] ファイルパスの数だけ，ファイルパスを読み込む
    
    Parameters
    ---------------
    filepaths : ndarray (1dim [ファイル数] : ファイルパスが1要素のオブジェクト配列)
        データが格納されたファイルのパス群
    skiprows : int
        データの読み飛ばす行 (デフォルトは0)
    usecols : リスト
        データの使用する列 (デフォルトはNone)

    Returns
    ---------------
    data_lst : np.float (2 dim [データ数, データ長])
        データ群
    
    """ 
    
    data_lst = list([])
    count = 0
    """ [1] ファイルパスの数だけ，ファイルパスを読み込む """
    for file in filepaths:
        data_tmp = pd.read_csv(file, sep=';', usecols=usecols, skiprows=skiprows, header=None)
        data_tmp = data_tmp.values
        
        data_lst.append(data_tmp)
        count += 1
        
    return data_lst


def load_data_from_filepaths(filepaths, use_entire_data=True, use_data_range=True, sample_rate=True, skiprows=0, usecols=None):
    """
    ファイルパス群からデータ群を入力として読み込む．

    [1] ファイルパスの数だけ，ファイルを読み込み
    [2] データの1部分を使用する場合の分岐処理 / 全データを使用する場合は，開始から整数時間だけ使い，残りの小数時間は切り捨てる．
    [3] データ格納
    
    Parameters
    ---------------
    filepaths : ndarray (1dim [ファイル数] : ファイルパスが1要素のオブジェクト配列)
        データが格納されたファイルのパス群
    use_entire_data : blool
        True : データの使用範囲を全体とする / False : 一部分とする．
    use_data_range : list (1dim, [2])
        use_entire_dataがFalseの時に，使用するデータ範囲を指定する．
    sample_rate : int
        データのサンプルレート
    skinpows : int
        データの読み飛ばす行 (デフォルトは0)
    usecols : list (1dim [2])
        データの使用する列 (デフォルトはNone:全使用)

    Returns
    ---------------
    data_lst : np.float (2 dim [データ数, データ長])
        データ群
    
    """ 
    
    data_lst = list([])  # データを格納するリスト
    count = 0

    """ [1] ファイルパスの数だけ，ファイルを読み込み """
    for file in filepaths:
        data_tmp = np.loadtxt(file, delimiter=',', skiprows=skiprows, usecols=usecols)
        
        """" [2] データの1部分を使用する場合の分岐処理 / 全データを使用する場合は，開始から整数時間だけ使い，残りの小数時間は切り捨てる． """
        if use_entire_data == False:
            data_lng = data_tmp.shape[-1]

            # データの次元が1or2以上で分岐
            # 表記が異なるだけで，処理自体は同様
            if data_tmp.ndim == 1:

                # 使用データ範囲がデータ長より短い場合は，使用データ範囲に従う．
                # 使用データ範囲がデータ長より長い場合は，データ長に従う．
                if use_data_range[1] * sample_rate <= data_lng:
                    data_tmp = data_tmp[use_data_range[0] * sample_rate : use_data_range[1] * sample_rate]
                else:
                    data_tmp = data_tmp[use_data_range[0] * sample_rate :]
            else:
            
                if use_data_range[1] * sample_rate <= data_lng: # 使用データ範囲がデータ長より短い場合
                    data_tmp = data_tmp[:, use_data_range[0] * sample_rate : use_data_range[1] * sample_rate]
                else:
                    data_tmp = data_tmp[:, use_data_range[0] * sample_rate :]

        """ [3] データ格納 """
        data_lst.append(data_tmp)
        count += 1
        
    return data_lst


def extract_filepaths_for_use(input_files, use_all_files, use_file_indxs, use_folder_for_input, input_folder, extension=''):
    """
    入力ファイルパス/フォルダパスの整理を行う．

    [1] 入力データの全てのファイルパスを取得 (入力がファイルかフォルダかに関わらず，最終的にはファイルパス群が出力される．)
    [2] 入力データのファイルパスの中で，使用するデータを取り出す．

    Parameters
    ---------------
    input_files : ndarray (1dim [ファイルパス数])
        入力ファイルパス群 / ファイルパスを1要素とするndarray
    use_all_files : bool
        True: input_filesのファイルを全て使用する． / False: use_file_indxsで指定するファイルを使用する．
    use_file_indxs : list (1dim [使用するファイルパス数])
        入力ファイルパス群の中で使用するファイルパスを指定するインデックス群
    use_folder_for_input : bool
        True: Inputとしてフォルダを指定する / False: Inputとしてファイルを指定する．
    input_folder : string
        use_folder_for_inputがTrueの場合，この変数で指定するフォルダ下のファイルを全て入力する．

    Returns
    ---------------
    use_filepaths : ndarray (1dim [使用するファイルパス数])
        使用するファイルパス群

    """

    """ [1] 入力データの全てのファイルパスを取得 (入力がファイルかフォルダかに関わらず，最終的にはファイルパス群が出力される．) """
    if use_folder_for_input == True:
        use_filepaths = glob.glob(input_folder + r'/*' + extension)
        use_filepaths = sorted(use_filepaths, key=natural_keys)
    else:
        use_filepaths = input_files

    """ [2] 入力データのファイルパスの中で，使用するデータを取り出す． """
    if use_all_files == False:
        use_filepaths = np.array(use_filepaths)
        use_filepaths = use_filepaths[use_file_indxs]
            
    return use_filepaths


if __name__ == '__main__':
   """ テスト用　"""
   
   use_filepaths = extract_filepaths_for_use(INPUT_FILES, USE_ALL_FILES, USE_FILE_INDXS, USE_FOLDER_FOR_INPUT, INPUT_FOLDER)
   data = load_data_from_filepaths(use_filepaths)
