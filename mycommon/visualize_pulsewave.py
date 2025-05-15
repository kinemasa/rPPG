import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import numpy as np
import os
def plot_pulse_wave_all(pulse_wave):
    """
    G成分の平均値を時系列で可視化する関数。
    """
    if  len(pulse_wave) > 0:
        frame_indices = list(range(len(pulse_wave)))
        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices, pulse_wave, linestyle='-', color='green')
        plt.title("Average Green Signal over Frames")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean Green Intensity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("可視化するデータがありません。")
        
        
def plot_pulse_wave(pulse_wave,sampling_rate,start_time,time,title,save_path):
    
    if  len(pulse_wave) > 0:
        min_time =start_time*sampling_rate
        max_time = time *sampling_rate
        pulse_wave=pulse_wave[min_time:]
        pulse_wave =pulse_wave[:max_time]
        frame_indices = list(range(len(pulse_wave)))
        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices, pulse_wave, linestyle='-', color='green')
        plt.title("Pulse wave"+title)
        plt.xlabel("Frame Index")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.tight_layout()
        # 保存パスが指定されていれば保存
        if save_path:
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"✅ グラフを保存しました: {save_path}")
    else:
        print("可視化するデータがありません。")
        
def read_pulse_csv(filepath):
    """CSVファイルを読み込んでDataFrameを返す"""
    try:
        df = pd.read_csv(filepath)
        if "time_sec" not in df.columns or "pulse" not in df.columns:
            raise ValueError("CSVに必要な列 'time_sec' または 'pulse' が見つかりません。")
        return df
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return None

def plot_pulse_wave_from_csv(df,sampling_rate=60,start_time=0,duration=None, title="Pulse Waveform"):
    """脈波データのグラフを描画"""
    
    if duration is not None:
        max_time = duration
        df = df[start_time<df["time_sec"] ]
        df = df[df["time_sec"] <= max_time]
    else:
        df = df[start_time<df["time_sec"] ]

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_sec"], df["pulse"], label="Pulse Wave", color='blue')
    plt.xlabel("Time (sec)")
    plt.ylabel("Pulse Amplitude")
    plt.title(f"{title} (0–{duration or 'All'} sec)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_multi_pulse_waves_from_csv(pulse_data_list, labels, start_time=0, duration=None,normalize=False):
    
    plt.figure(figsize=(10, 4))
    y_min, y_max = float('inf'), float('-inf')

    for i, df in enumerate(pulse_data_list):
        time = df['time_sec'].values
        pulse = df['pulse'].values
        
        if normalize:
            # min-max 正規化 [-0.10, 0.10]
            pulse = (pulse - np.min(pulse)) / (np.max(pulse) - np.min(pulse))  # [0, 1]
            pulse = pulse * 0.20 - 0.10  # [0, 1] → [-0.10, 0.10]
        plt.plot(time, pulse, label=labels[i])

    plt.xlabel("Time (sec)")
    plt.ylabel("Pulse Amplitude")
    title_range = f"{start_time}–{start_time + duration} sec" if duration is not None else "All"
    plt.title(f"Pulse Waves ({title_range})")
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.grid(True)
    plt.ylim(-0.11, 0.11) if normalize else None
    plt.tight_layout()
    plt.show()

