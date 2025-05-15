import sys
import os
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## 自作ライブラリ
from mycommon.select_folder import select_file
from mycommon.road_and_save import save_pulse_to_csv,read_pulse_csv
from mycommon.visualize_pulsewave import plot_pulse_wave
from mycommon.processing_pulse import detrend_pulse,sg_filter_pulse,bandpass_filter_pulse


def main():
    
    current_path = Path(__file__)
    parent_path =current_path.parent
    saved_folder = str(parent_path)+"\\saved_pulse\\"
    saved_subfolder =saved_folder +"\\subject1\\"
    
    sampling_rate = 60
    bandpath_width = [0.75,3.0]
    start_time =0
    time = 30
    
    input_file = select_file()
    df = read_pulse_csv(input_file)
    pulse_wave = df["pulse"]
    pulse_wave = np.asarray(pulse_wave)
    detrend_pulsewave = detrend_pulse(pulse_wave,sampling_rate)
    bandpass_pulsewave = bandpass_filter_pulse(detrend_pulsewave,bandpath_width,sampling_rate)
    sg_filter_pulsewave =sg_filter_pulse(bandpass_pulsewave,sampling_rate)
    
    plot_pulse_wave(pulse_wave,sampling_rate,start_time,time)
    plot_pulse_wave(detrend_pulsewave,sampling_rate,start_time,time)
    plot_pulse_wave(bandpass_pulsewave,sampling_rate,start_time,time)
    plot_pulse_wave(sg_filter_pulsewave,sampling_rate,start_time,time)
    

    save_pulse_to_csv(pulse_wave,saved_subfolder+"pulsewave.csv",60)
    save_pulse_to_csv(detrend_pulsewave,saved_subfolder+"detrend_pulse.csv",60)
    save_pulse_to_csv(bandpass_pulsewave,saved_subfolder+"bandpass_pulse.csv",60)
    save_pulse_to_csv(sg_filter_pulsewave,saved_subfolder+"sgfilter_pulse.csv",60)
    
if __name__ =="__main__":
    main()
    