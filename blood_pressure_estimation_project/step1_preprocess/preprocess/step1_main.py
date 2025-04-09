"""
脈波信号を読み込み処理を行う関数
"""

from common.imports import np, plt, signal, glob, os, Path, natsorted
from common.pulse_preprocess import bandpass_filter_pulse, detect_pulse_peak,detrend_signal,extract_green_signal
from common.visualize import visualize_pulse


from common.config import (
    DEFAULT_BANDPASS_RANGE_HZ,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_CAPTURE_TIME,
    DEFAULT_OUTPUT_FOLDER_NAME
)


def process_csv_file(csv_path,save_dir, sampling_rate= DEFAULT_SAMPLING_RATE, time = DEFAULT_CAPTURE_TIME):
    """1ファイルに対して脈波処理・保存・可視化を実行"""
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    # G成分抽出と保存
    pulse_raw = extract_green_signal(csv_path)
    

    # 傾き除去とフィルタ処理
    pulse_detrended = detrend_signal(pulse_raw)
    pulse_filtered = bandpass_filter_pulse(pulse_detrended, DEFAULT_BANDPASS_RANGE_HZ, sampling_rate)

    # ピーク検出
    peak_indexes, valley_indexes = detect_pulse_peak(pulse_filtered, sampling_rate)

    # 可視化と保存
    save_img_peak = os.path.join(save_dir, "g_peak.png")
    visualize_pulse(pulse_filtered, save_img_peak, peak_indexes, valley_indexes, sampling_rate, time)
    
    np.savetxt(os.path.join(save_dir, "g.csv"), pulse_filtered, delimiter=",")


def run_step1_all(input_dir, save_dir, sampling_rate= DEFAULT_SAMPLING_RATE, time= DEFAULT_CAPTURE_TIME):
    """フォルダ内のすべてのCSVファイルに対して処理を実行"""
    csv_files = natsorted(glob.glob(os.path.join(input_dir, "*.csv")))
    print(len(csv_files))
    for csv_file in csv_files:
        process_csv_file(csv_file, save_dir, sampling_rate, time)
        print(f"{csv_file} is Done!")