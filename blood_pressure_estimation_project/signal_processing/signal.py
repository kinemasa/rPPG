import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy import signal

def extract_green_signal(csv_path) :
    pulse_data = np.loadtxt(csv_path, delimiter=",")
    return pulse_data

def bandpass_filter_pulse(pulse_detrended, bandpass_filter_range_hz, sampling_rate):
    # バンドパスフィルタ処理
    nyq = 0.5 * sampling_rate
    b, a = signal.butter(1, [bandpass_filter_range_hz[0] / nyq, bandpass_filter_range_hz[1] / nyq], btype='band')
    pulse_bandpass_filtered = signal.filtfilt(b, a, pulse_detrended)

    return pulse_bandpass_filtered


def detrend_signal(signal_data: np.ndarray):
    #傾き除去
    return signal.detrend(signal_data)