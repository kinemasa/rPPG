import tkinter as tk
from tkinter import filedialog
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mycommon.visualize_pulsewave import read_pulse_csv ,plot_pulse_wave_from_csv
from mycommon.processing_pulse import calculate_snr
from mycommon.select_folder import select_file

def main():
    
    sampling_rate =60
    duration = None
    OUTPUT_FOLDER ="c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\main-folder\\" 
    # ファイル選択ダイアログ
    file_path = select_file()
    if not file_path:
        print("ファイルが選択されませんでした。")
        return

    # 読み込み＆プロット
    df = read_pulse_csv(file_path)
    
    if df is not None:
        pulse = df["pulse"]
        snr = calculate_snr(pulse,sampling_rate,OUTPUT_FOLDER)
        print(snr)

if __name__ == "__main__":
    main()