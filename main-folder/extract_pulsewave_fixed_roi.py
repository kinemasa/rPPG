import sys
import os
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## 自作ライブラリ
from mycommon.select_folder import select_folder
from mycommon.road_and_save import get_sorted_image_files,get_sorted_image_100files,save_pulse_to_csv
from mycommon.extract_pulsewave import select_roi,compute_green_means
from mycommon.visualize_pulsewave import plot_pulse_wave
from mycommon.processing_pulse import detrend_pulse,sg_filter_pulse,bandpass_filter_pulse


def main():
    
    current_path = Path(__file__)
    parent_path =current_path.parent
    saved_folder = str(parent_path)+"\\saved_pulse\\"
    saved_subfolder =saved_folder +"\\subject1\\"
    sampling_rate = 60
    bandpath_width = [0.75,3.0]
    time = 5
    frame = sampling_rate*time
    input_folder = select_folder()
    input_image_paths = get_sorted_image_files(input_folder)
    roi = select_roi(input_image_paths[0])
    
    pulse_wave = compute_green_means(input_image_paths,roi)
    pulse_wave = np.asarray(pulse_wave)
    detrend_pulsewave = detrend_pulse(pulse_wave,sampling_rate)
    
    
    bandpass_pulsewave = bandpass_filter_pulse(detrend_pulsewave,bandpath_width,sampling_rate)
    sg_filter_pulsewave =sg_filter_pulse(bandpass_pulsewave,sampling_rate)
    
    plot_pulse_wave(pulse_wave,frame)
    plot_pulse_wave(detrend_pulsewave,frame)
    plot_pulse_wave(bandpass_pulsewave,frame)
    plot_pulse_wave(sg_filter_pulsewave,frame)
    
    save_pulse_to_csv(pulse_wave,saved_subfolder+"pulsewave.csv",60)
    save_pulse_to_csv(detrend_pulsewave,saved_subfolder+"detrend_pulse.csv",60)
    save_pulse_to_csv(bandpass_pulsewave,saved_subfolder+"bandpass_pulse.csv",60)
    save_pulse_to_csv(sg_filter_pulsewave,saved_subfolder+"sgfilter_pulse.csv",60)
    
if __name__ =="__main__":
    main()
    