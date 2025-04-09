import numpy as np

def extract_green_signal(csv_path) :
    pulse_data = np.loadtxt(csv_path, delimiter=",")
    return pulse_data