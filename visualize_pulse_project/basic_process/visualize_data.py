"""
データ可視化

20200821 Kaito Iuchi
"""


import glob
import os
import sys

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

sys.path.append('/Users/ik/GglDrv2/FY-2021/02_Collaboration/DaikinIndustry/03_Presentation/202108_forDelivery/skinvideo_to_bloodpressure')  # 自作ライブラリを格納しているフォルダのパスを通す．
import basic_process.make_output_folder as mof  # 出力データを格納するフォルダを作成するためのモジュール
import basic_process.basic_loading_process as blp  # 基本的なデータ入力処理のためのモジュール
import basic_process.basic_pulse_process as bpp  # 基本的な脈波処理のためのモジュール
import basic_process.basic_timeseries_process as btp  # 基本的な時系列処理のためのモジュール
import basic_process.basic_ndlst_process as bnp  # ndarrayとリストの相互変換のためのモジュール
import basic_process.visualize_data as vd  # 基本的なデータ可視化処理のためのモジュール


def visualize_blandaltman(axes, md_nd, mn_nd, lw=1.5,
                          alpha=0.5, color=cm.winter,
                          save_fig=False, output_folder='', fig_tag=''):
    """
    時系列データ(脈波を想定)を可視化する．

    Parameters
    ---------------
    axes : matplotlib.axes
        matplotlib.axesオブジェクト
    crr_nd : np.ndarray (1dim / [データ長])
        相関係数群
    lw : np.float
        線のサイズ
    alpha : np.float
        線の透明度
    color : matplotlib.cm
        matplotlib.cmオブジェクト
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing

    """
    seaborn.set_style("darkgrid")
    
    data_lng = md_nd.shape[0]
    
    axes.scatter(mn_nd, md_nd, lw=lw, alpha=alpha, marker= '.', color=cm.viridis(0.3), s=25)
    # axes.set_xlim([100, 150])
    axes.set_ylim([100, -100])
    
    if save_fig == True:
        plt.savefig(output_folder + '/Bland-Altman-' + fig_tag + '.png')


def visualize_crr(axes, crr_nd,
                  alpha=0.5, color=cm.winter,
                  save_fig=False, output_folder='', fig_tag=''):
    """
    時系列データ(脈波を想定)を可視化する．

    Parameters
    ---------------
    axes : matplotlib.axes
        matplotlib.axesオブジェクト
    crr_nd : np.ndarray (1dim / [データ長])
        相関係数群
    lw : np.float
        線のサイズ
    alpha : np.float
        線の透明度
    color : matplotlib.cm
        matplotlib.cmオブジェクト
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing

    """
    seaborn.set_style("darkgrid")
    
    data_lng = crr_nd.shape[0]
    
    x = np.linspace(1, data_lng + 1, data_lng)
    axes.barh(x, crr_nd, alpha=alpha, color=cm.winter(0.7))
    axes.set_xlim([-1, 1])
    
    if save_fig == True:
        plt.savefig(output_folder + '/CorrelationCoefficient-' + fig_tag + '.png')


def visualize_bp(axes, sbp, dbp, sbp2, dbp2,
                 lw=3, alpha=1.0, color=cm.winter,
                 save_fig=False, output_folder='', fig_tag=''):
    """
    時系列データ(脈波を想定)を可視化する．

    Parameters
    ---------------
    axes : matplotlib.axes
        matplotlib.axesオブジェクト
    pulse1 : np.ndarray (1dim / [データ長])
        BP脈波
    pulse2 : np.ndarray (1dim / [データ長])
        SGフィルタ脈波
    pulse3 : np.ndarray (1dim / [データ長])
        デトレンド脈波
    peak1_indx: np.ndarray (1dim / [データ長])
        上側ピーク
    peak2 : np.ndarray (1dim / [データ長])
        下側ピーク
    lw : np.float
        線のサイズ
    alpha : np.float
        線の透明度
    color : matplotlib.cm
        matplotlib.cmオブジェクト
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing

    """
    seaborn.set_style("darkgrid")
    
    x = np.linspace(0, sbp.shape[0], sbp.shape[0])
    axes.scatter(x, sbp, lw=lw, alpha=alpha, marker= '.', color=cm.viridis(0.3), s=50)
    x = np.linspace(0, dbp.shape[0], dbp.shape[0])
    axes.scatter(x, dbp, lw=lw, alpha=alpha, marker= '.', color=cm.viridis(0.6), s=50)
    x = np.linspace(0, sbp2.shape[0], sbp2.shape[0])
    axes.plot(x, sbp2, color=cm.winter(0.3), lw=lw, alpha=alpha)
    x = np.linspace(0, dbp2.shape[0], dbp2.shape[0])
    axes.plot(x, dbp2, color=cm.winter(0.6), lw=lw, alpha=alpha)

    axes.axis('tight')
    
    if save_fig == True:
        plt.savefig(output_folder + '/BP-' + fig_tag + '.png')


def visualize_failed_flag(failed_flag_lst, sample_rate, save_fig, output_folder, fig_tag):
    """
    秒ごとのエラーフラグの量の可視化

    Parameters
    ---------------
    failed_flag_lst : リスト (1dim [データ数] : np.ndarray (1dim / [データ長]))
        エラーフラグ
    sample_rate : np.int
        データのサンプルレート
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing
    
    """
    
    seaborn.set_style("darkgrid")
    
    data_num = len(failed_flag_lst)
    
    # fig = plt.figure(figsize=(12,9))
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)
    
    # data = failed_flag_lst[0]
    # data_lng = data.shape[0]
    # success_num = np.count_nonzero(data)
    # failure_num = data_lng - success_num
    # color = [cm.winter(0.3), cm.winter(0.6)]
    # ax1.pie([success_num, failure_num], autopct='%1.1f%%', colors=color)
    
    # data = failed_flag_lst[1]
    # data_lng = data.shape[0]
    # success_num = np.count_nonzero(data)
    # failure_num = data_lng - success_num
    # color = [cm.winter(0.3), cm.winter(0.6)]
    # ax2.pie([success_num, failure_num], autopct='%1.1f%%', colors=color)
    
    # data = failed_flag_lst[2]
    # data_lng = data.shape[0]
    # success_num = np.count_nonzero(data)
    # failure_num = data_lng - success_num
    # color = [cm.winter(0.3), cm.winter(0.6)]
    # ax3.pie([success_num, failure_num], autopct='%1.1f%%', colors=color)
    
    # data = failed_flag_lst[3]
    # data_lng = data.shape[0]
    # success_num = np.count_nonzero(data)
    # failure_num = data_lng - success_num
    # color = [cm.winter(0.3), cm.winter(0.6)]
    # ax4.pie([success_num, failure_num], autopct='%1.1f%%', colors=color)
    
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, ])
    for num in range(data_num):
        data = failed_flag_lst[num]
        data_lng = data.shape[0]
        data_cnt = np.empty([data_lng])
        for a in range(data_lng):
            data_cnt[a] = np.count_nonzero(data[:a] == 0)
        data_cnt = data_cnt / data_lng
        x = np.linspace(0, data_lng, data_lng)
        ax.plot(x, data_cnt, color=cm.winter(num/data_num), lw=1.2, alpha=1.0)
        
    if save_fig == True:
        plt.savefig(output_folder + '/FailedFlag-' + fig_tag + '.png')
       
    return None


def visualize_pulse(axes, pulse1, pulse2=False, pulse3=False, pulse4=False, peak1_indx=False, peak2_indx=False,
                    lw=1, alpha=0.6, color=cm.winter,
                    save_fig=False, output_folder='', fig_tag=''):
    """
    時系列データ(脈波を想定)を可視化する．

    Parameters
    ---------------
    axes : matplotlib.axes
        matplotlib.axesオブジェクト
    pulse1 : np.ndarray (1dim / [データ長])
        BP脈波
    pulse2 : np.ndarray (1dim / [データ長])
        SGフィルタ脈波
    pulse3 : np.ndarray (1dim / [データ長])
        デトレンド脈波
    peak1_indx: np.ndarray (1dim / [データ長])
        上側ピーク
    peak2 : np.ndarray (1dim / [データ長])
        下側ピーク
    lw : np.float
        線のサイズ
    alpha : np.float
        線の透明度
    color : matplotlib.cm
        matplotlib.cmオブジェクト
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing

    """
    seaborn.set_style("darkgrid")
    
    data_lng = pulse1.shape[0]
    
    # 脈波データを可視化
    # fig1_ax2 = fig1_ax1.twinx()
    # fig1_ax2.plot(x, pulse[:used_length], color=cm.winter(0.1),linewidth=0.3, alpha=0.3)
    
    
    if pulse4 is not False: 
        x = np.linspace(0, pulse4.shape[0], pulse4.shape[0])
        axes.plot(x, pulse4, color=color(0.2), lw=lw, alpha=alpha)
    if pulse3 is not False: 
        x = np.linspace(0, pulse3.shape[0], pulse3.shape[0])
        axes.plot(x, pulse3, color=color(0.4), lw=lw, alpha=alpha)
    if pulse2 is not False:
        x = np.linspace(0, pulse2.shape[0], pulse2.shape[0])
        axes.plot(x, pulse2, color=color(0.6), lw=lw, alpha=alpha)
    x = np.linspace(0, pulse1.shape[0], pulse1.shape[0])
    axes.plot(x, pulse1, color=color(0.8), lw=lw, alpha=alpha)
    
    if peak1_indx is not False:
        axes.scatter(x[peak1_indx], pulse1[peak1_indx], marker= '+', color=cm.viridis(0.3), s=100)
    if peak2_indx is not False:
        axes.scatter(x[peak2_indx], pulse1[peak2_indx], marker='+', color=cm.viridis(0.3), s=100)
    axes.axis('tight')
    # fig1_ax1.xlabel('Time [s]')
    # fig1_ax1.ylabel('Amplitude')

    axes.set_ylim([50, 150])

    if save_fig == True:
        plt.savefig(output_folder + '/BP-' + fig_tag + '.png')


def visualize_normalized_pulse(axes, data, peak_indx=0, lw=0.1, alpha=0.8, color=cm.ocean,
                               save_fig=False, output_folder='', fig_tag=''):
    """
    時系列データ(正規化脈波を想定)を可視化する．

    Parameters
    ---------------
    axes : matplotlib.axes
        matplotlib.axesオブジェクト
    data : np.ndarray (2dim / [データ数, データ長])
        時系列データ
    lw : np.float
        線のサイズ
    alpha : np.float
        線の透明度
    color : matplotlib.cm
        matplotlib.cmオブジェクト
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing

    """
        
    seaborn.set_style("darkgrid")
    
    # 正規化1脈動分脈波の窓での平均化されたものの可視化
    data_num = data.shape[0]
    data_lng = data.shape[1]
    
    x = np.linspace(0, 1, data_lng)
    for num in range(data_num):
        axes.plot(x, data[num], lw=lw, alpha=alpha, color=color(num/(data_num * 1.15)))
        
    global a, b
    a = peak_indx
    b = data
    if isinstance(peak_indx, list):
        for num in range(data_num):
            indx = peak_indx[num]
            axes.scatter(x[indx], data[num, indx], marker= '+', color=cm.viridis(0.5), s=10, alpha=0.5)
        
    if save_fig == True:
        plt.savefig(output_folder + '/NormalizedPulse-' + fig_tag + '.png')


def visualize_data(axes, data, lw=0.8, alpha=0.8, color=cm.winter,
                   save_fig=False, output_folder='', fig_tag=''):
    """
    時系列データを可視化する．

    Parameters
    ---------------
    axes : matplotlib.axes
        matplotlib.axesオブジェクト
    data : np.ndarray (2dim / [データ数, データ長])
        時系列データ
    lw : np.float
        線のサイズ
    alpha : np.float
        線の透明度
    color : matplotlib.cm
        matplotlib.cmオブジェクト
    save_fig : bool
        グラフを保存するかどうかのフラグ | True: 保存する． / False: 保存しない．
    output_folder : string
        出力フォルダ
    fig_tag : string
        出力ファイルタグ

    Returns
    ---------------
    Nothing

    """
    
    seaborn.set_style("darkgrid")
    
    # 特徴量を可視化
    if data.ndim > 1:
        data_num = data.shape[0]
        data_lng = data.shape[1]
        x = np.linspace(0, data_lng, data_lng)
        for num in range(data_num):
            axes.plot(x, data[num, :], lw=lw, alpha=alpha, color=color(num/data_num))
    else:
        data_lng = data.shape[0]
        x = np.linspace(0, data_lng, data_lng)
        axes.plot(x, data[:], lw=lw, alpha=alpha, color=color(0.5))

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Time [sec]", fontsize=18)
    plt.ylabel("Amplitude [Arb. Unit]", fontsize=18)

    # axes.set_ylim([-3.5, 3.5])
    # ax.set_xlim([0, 1])

    if save_fig == True:
        plt.savefig(output_folder + '/Data-' + fig_tag + '.png')


if __name__ == '__main__':

    print('\r\n\nThis program has been finished successfully!')
    
    