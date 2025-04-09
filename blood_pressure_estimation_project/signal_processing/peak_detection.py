from scipy import signal

def detect_pulse_peak(pulse_bandpass_filtered, sampling_rate):
    # ピーク検出
    peak_indexes = signal.argrelmax(pulse_bandpass_filtered, order=int(sampling_rate / 3.0))[0]
    valley_indexes = signal.argrelmin(pulse_bandpass_filtered, order=int(sampling_rate / 3.0))[0]

    return peak_indexes, valley_indexes