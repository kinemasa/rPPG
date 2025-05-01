import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy import signal
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def pulse_csv_to_png_5s(dir, pulse_data, name,sample_rate):
    # 脈波可視化用関数(5秒)
    save_graph_img_name = dir + name + "_5s.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pulse_data[:sample_rate *5])
    plt.xticks([0, sample_rate, sample_rate*2, sample_rate*3, sample_rate*4, sample_rate*5])
    ax.grid(linestyle="--", color="gray")
    plt.savefig(save_graph_img_name)
    plt.close()
    
def pulse_csv_to_png(dir, pulse_data, name,sample_rate,time):
    # 脈波可視化用関数
    save_graph_img_name = dir + name + ".png"
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pulse_data[:sample_rate *time])
    plt.xticks([0, sample_rate*time])
    ax.grid(linestyle="--", color="gray")
    plt.savefig(save_graph_img_name)
    plt.close()

def visualize_pulse(pulse_bandpass_filtered, filename_save_img_peak, peak_indexes, valley_indexes,sample_rate,time):
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    x = np.linspace(0, pulse_bandpass_filtered.shape[0], pulse_bandpass_filtered.shape[0])
    ax.plot(x, pulse_bandpass_filtered)

    # ax.scatter(x[peak_indexes], pulse[peak_indexes], marker='x', color='green', s=150)
    ax.scatter(x[valley_indexes], pulse_bandpass_filtered[valley_indexes], marker='x', color='green', s=150)
    # plt.xticks([0, 300, 600, 900, 1200, 1500, 1800])
    plt.xticks([0, sample_rate*time])
    ax.grid(linestyle="--", color="gray")
    plt.savefig(filename_save_img_peak)
    plt.close()