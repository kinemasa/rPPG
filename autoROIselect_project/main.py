import os
import glob
import numpy as np
import sys
from makeskinseparation.make_skinseparation import makeSkinSeparation 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mycommon.select_folder import select_folder
def main():
    # 入力ディレクトリ
    INPUT_DIR = select_folder()
    # 出力ディレクトリ
    OUTPUT_DIR = INPUT_DIR +"\\result\\" # ← 出力先フォルダ

    # 入力画像の拡張子に応じてファイルリスト取得
    input_image_list = sorted(
        glob.glob(os.path.join(INPUT_DIR, '*.png')) +  # 必要に応じて変更
        glob.glob(os.path.join(INPUT_DIR, '*.jpg')) +
        glob.glob(os.path.join(INPUT_DIR, '*.CR2')) +
        glob.glob(os.path.join(INPUT_DIR, '*.npy'))
    )

    # メラニン・ヘモグロビン・陰影のベクトル
    melanin    =[0.2203, 0.4788, 0.8499]
    hemoglobin =[0.4350, 0.6929, 0.5750]
    shading    =[ 1.0000, 1.0000, 1.0000 ]

    vector = [melanin,hemoglobin,shading]

    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 分離処理の実行
    makeSkinSeparation(INPUT_DIR, input_image_list, OUTPUT_DIR, vector)

if __name__ == '__main__':
    main()