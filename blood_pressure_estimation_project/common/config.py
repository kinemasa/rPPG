# common/config.py


# バンドパスフィルタの通過帯域（Hz）
DEFAULT_BANDPASS_RANGE_HZ = [0.75, 4.0]
# サンプリングレート（fps）
DEFAULT_SAMPLING_RATE = 60
# 撮影時間（秒）
DEFAULT_CAPTURE_TIME = 60
# 出力フォルダ名（ファイル保存時などに使用）
DEFAULT_OUTPUT_FOLDER_NAME = ""

## step2の処理
RESAMPLING_RATE = 256  # アップサンプリンする際のフレームレート
WINDOW_SIZE = 10  # 波形の平均化処理に用いる波形の個数
FEATURES_NUM_CN = 29  # 脈波概形特徴量の個数
FEATURES_NUM_DR = 22  # 脈波導関数特徴量の個数
upper_ratio = 0.10
upper_ratio_g = 0.10
upper_ratio_nir = 0.20