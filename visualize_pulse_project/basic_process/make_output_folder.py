"""
日付フォルダを作成するためのモジュール

20190516 Kaito Iuchi
"""


""" 標準ライブラリのインポート """
import os
from datetime import datetime


def make_output_folder(folder, program_name):
    """
    入力されたフォルダに本日の日付のフォルダを作成する．

    [1] 本日の日付を取得
    [2] 出力フォルダに，本日の日付がラベリングされたフォルダを作成 (複数回実行しても上書きされないように，重複をナンバリングで解消)

    Parameters
    ---------------
    folder : string
        日付フォルダを作成するフォルダ
    program_name : string
        プログラムの名前

    Returns
    ---------------
    Nothing

    """

    """ [1] 本日の日付を取得 """
    dt = datetime.now()
    datestr = dt.strftime('%Y%m%d')

    """ [2] 出力フォルダに，本日の日付がラベリングされたフォルダを作成 (複数回実行しても上書きされないように，重複をナンバリングで解消) """
    k = 0
    flag = 0
    date_folder = ''
    while flag == 0:
        date_folder = folder + '/' + '[' + program_name + ']-' + datestr + '_' + str(k).zfill(2)
        if not os.path.exists(date_folder):
            os.mkdir(date_folder)
            flag = 1
        k += 1

    print(f'\nThe output folder has been created successfully!')

    return date_folder
