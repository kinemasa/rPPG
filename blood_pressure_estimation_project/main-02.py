import os
import numpy as np
from pathlib import Path
from scipy import signal

from common.utils import select_folder

from blood_pressure_estimation_project.signal_processing.signal import bandpass_filter_pulse
from signal_processing.peak_detection import detect_pulse_peak
from signal_processing.get_feature import calc_contour_features,calc_dr_features
from signal_processing.analyze import analyze_pulses,select_pulses_by_statistics,upsample_data
from signal_processing.t_generation import generate_t1,generate_t2
from common.read_yaml import load_config

current_path = Path(__file__)
parent_path = current_path.parent   
config = load_config(str(parent_path)+"\\config\\config_0102.yaml")


DEFAULT_BANDPASS_RANGE_HZ = config["preprocessing"]["bandpass_range_hz"]
DEFAULT_SAMPLING_RATE     = config["preprocessing"]["sampling_rate"]
DEFAULT_CAPTURE_TIME      = config["preprocessing"]["capture_time"]
DEFAULT_OUTPUT_FOLDER_NAME = config["preprocessing"]["output_folder_name"]
RESAMPLING_RATE  =config["feature_extraction"]["resampling_rate"]

def process_csv_file(csv_file,sample_rate =DEFAULT_SAMPLING_RATE):
    
    pulse_bandpass_filtered = np.loadtxt(csv_file, delimiter=",")
    
    # 傾き除去
    pulse_detrended = signal.detrend(pulse_bandpass_filtered)
    # バンドパスフィルタ処理
    pulse_bandpass_filtered = bandpass_filter_pulse(pulse_detrended,DEFAULT_BANDPASS_RANGE_HZ , DEFAULT_SAMPLING_RATE)
    
    # ピーク検出 (上側，下側)
    
    peak_indexes, valley_indexes = detect_pulse_peak(pulse_bandpass_filtered, sample_rate)
    
    # 各1波形の面積，持続時間，最大振幅を格納するリスト
    area_list = []
    duration_time_list = []
    amplitude_list = []
    acceptable_idx_list = []

    ## 脈波波形の微分等の解析
    area_list, duration_time_list, amplitude_list, acceptable_idx_list,pulse_waveform_num= analyze_pulses(pulse_bandpass_filtered,valley_indexes)
    acceptable_idx_list =select_pulses_by_statistics(area_list, duration_time_list, amplitude_list, pulse_waveform_num)
    
    ## t1を求める
    t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, success = generate_t1(pulse_bandpass_filtered,valley_indexes,amplitude_list,acceptable_idx_list,RESAMPLING_RATE)
    
    ## t2を求める
    t2 =generate_t2(t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, upper_ratio=0.10)
    
    # t2から特徴量を求める
    features_cn_array = calc_contour_features(t2, RESAMPLING_RATE)
    features_dr_array = calc_dr_features(t2, RESAMPLING_RATE)
    
    
    
    return features_cn_array,features_dr_array
    

def run_all(dir_csv_file,dir_blood_csv, save_dir, sampling_rate= DEFAULT_SAMPLING_RATE, time= DEFAULT_CAPTURE_TIME):
    """フォルダ内のすべてのCSVファイルに対して処理を実行"""
    csv_files = list(Path(dir_csv_file).glob("*.csv"))
    for csv_path in csv_files:
        features_cn_array,features_dr_array =process_csv_file(csv_path,sampling_rate)
        csv_stem = csv_path.stem  # ファイル名（拡張子なし）
        # 特徴量保存用のディレクトリ名の指定
        dir_name_save = dir_csv_file  + "\\"+csv_stem+"\\" + "features\\"
        os.makedirs(dir_name_save, exist_ok=True)

        # 特徴量保存用のファイル名の指定
        filename_save_features_cn = dir_name_save + "features_cn" + ".csv"
        filename_save_features_dr = dir_name_save + "features_dr" + ".csv"

        # 特徴量の保存
        np.savetxt(filename_save_features_cn, features_cn_array, delimiter=",")
        np.savetxt(filename_save_features_dr, features_dr_array, delimiter=",")
        print(csv_path, "is Done!")

def main():
    input_dir = select_folder("ミニカメラで取得したcsvファイルが入ったフォルダーを選択してください")
    output_dir = os.path.join(input_dir, DEFAULT_OUTPUT_FOLDER_NAME)
    run_all(input_dir, output_dir, DEFAULT_SAMPLING_RATE)


if __name__ == "__main__":
    main()

