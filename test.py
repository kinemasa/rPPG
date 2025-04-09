import numpy as np
import pandas as pd

# サンプリング設定
sampling_rate = 60  # 30Hz（1秒間に30サンプル）
duration = 30       # 10秒間のデータ
t = np.linspace(0, duration, sampling_rate * duration)

# 仮の脈波信号（正弦波＋ノイズ）
heart_rate_hz = 1.5  # 約72 bpm
# signal = 0.5 * np.sin(2 * np.pi * heart_rate_hz * t) + 0.05 * np.random.randn(len(t))
signal = 0.5 * np.sin(2 * np.pi * heart_rate_hz * t) 

# DataFrameにまとめる
df = pd.DataFrame({
    'pulse_signal': signal
})

# CSVとして保存
df.to_csv('pulse_signal.csv', index=False,header=None)

print("CSVファイル 'pulse_signal.csv' を出力しました。")