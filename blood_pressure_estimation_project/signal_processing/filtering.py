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


def detect_pulse_peak(pulse_bandpass_filtered, sampling_rate):
    # ピーク検出
    peak_indexes = signal.argrelmax(pulse_bandpass_filtered, order=int(sampling_rate / 3.0))[0]
    valley_indexes = signal.argrelmin(pulse_bandpass_filtered, order=int(sampling_rate / 3.0))[0]

    return peak_indexes, valley_indexes

def detrend_signal(signal_data: np.ndarray):
    #傾き除去
    return signal.detrend(signal_data)