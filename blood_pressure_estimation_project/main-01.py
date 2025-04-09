import os
import numpy as np
from pathlib import Path
from scipy import signal
from common.utils import select_folder
from signal_processing.csv_to_signal import extract_green_signal
from signal_processing.filtering import detrend_signal,bandpass_filter_pulse
from signal_processing.peak_detection import detect_pulse_peak
from common.visualize import visualize_pulse
from common.read_yaml import load_config

current_path = Path(__file__)
parent_path = current_path.parent     
config = load_config(str(parent_path)+"\\config\\config.yaml")

DEFAULT_BANDPASS_RANGE_HZ = config["preprocessing"]["bandpass_range_hz"]
DEFAULT_SAMPLING_RATE     = config["preprocessing"]["sampling_rate"]
DEFAULT_CAPTURE_TIME      = config["preprocessing"]["capture_time"]


def process_csv_file(csv_path, save_dir, sampling_rate=DEFAULT_SAMPLING_RATE, time=DEFAULT_CAPTURE_TIME):
    """1ファイルに対して脈波処理・保存・可視化を実行"""
    os.makedirs(save_dir, exist_ok=True)

    # G成分抽出
    pulse_raw = extract_green_signal(csv_path)

    # 傾き除去 & バンドパスフィルタ
    pulse_detrended = detrend_signal(pulse_raw)
    pulse_filtered = bandpass_filter_pulse(pulse_detrended, DEFAULT_BANDPASS_RANGE_HZ, sampling_rate)

    # ピーク検出
    peak_indexes, valley_indexes = detect_pulse_peak(pulse_filtered, sampling_rate)

    # 可視化 & 保存
    save_img_peak = os.path.join(save_dir, "g_peak.png")
    visualize_pulse(pulse_filtered, save_img_peak, peak_indexes, valley_indexes, sampling_rate, time)

    # フィルタ済み脈波を保存
    np.savetxt(os.path.join(save_dir, "g.csv"), pulse_filtered, delimiter=",")


def run_all():
    """入力フォルダ内のすべてのCSVファイルに処理を実行"""
    input_folder = select_folder("脈波CSVファイルのフォルダを選択してください")
    csv_files = list(Path(input_folder).glob("*.csv"))

    if not csv_files:
        print("CSVファイルが見つかりませんでした。")
        return

    for csv_path in csv_files:
        csv_stem = csv_path.stem
        output_folder = os.path.join(input_folder, csv_stem, "processed")
        process_csv_file(csv_path, output_folder)
        print(f"{csv_path} is Done !!")


if __name__ == "__main__":
    run_all()