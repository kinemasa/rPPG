import tkinter as tk
from tkinter import filedialog
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mycommon.visualize_pulsewave import read_pulse_csv ,plot_multi_pulse_waves_from_csv
from mycommon.select_folder import select_file
def main():
    
    sampling_rate =60
    start_time = 2
    duration = 2
    num_files = 3
    normalize = True
    pulse_data_list = []
    labels = []
    
    for i in range(num_files):
        file_path = select_file(f"{i+1}個目のファイルを選択してください")
        
        if not file_path:
            print("ファイルが選択されませんでした。")
            return
        
        df = read_pulse_csv(file_path)
        if df is not None:
            # 指定した時間範囲で切り出し
            df_trimmed = df[(df['time_sec'] > start_time) & (df['time_sec'] <= start_time + duration)]
            pulse_data_list.append(df_trimmed)
            folder_name = os.path.basename(os.path.dirname(file_path))
            labels.append(folder_name)
    

    plot_multi_pulse_waves_from_csv(pulse_data_list, labels, start_time, duration,normalize)

if __name__ == "__main__":
    main()