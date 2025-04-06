import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
#========================================================#
# 可視化用関数#
#=======================================================
def plot_histogram(data, title="Histogram", xlabel="Value", ylabel="Frequency", bins=100, color="blue", alpha=0.7):
    """
    データのヒストグラムをプロットする。

    Parameters:
        data (numpy.array): ヒストグラムに使用するデータ（1次元配列）。
        title (str): グラフのタイトル。
        xlabel (str): X軸のラベル。
        ylabel (str): Y軸のラベル。
        bins (int): ヒストグラムの棒の数。
        color (str): 棒の色。
        alpha (float): 棒の透明度。
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, range=(-2, 1), color=color, alpha=alpha)  # 範囲を -2 から 3 に指定
    plt.ylim(0,50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
def plot_3d_skinFlat_plotly(skin_flat, melanin_vector, hemoglobin_vector, vec, max_points=5000):
    """
    3次元プロットをPlotlyでインタラクティブに表示

    Parameters:
        skin_flat: (3, height, width) のnumpy配列。
        melanin_vector: メラニンベクトル。
        hemoglobin_vector: ヘモグロビンベクトル。
        vec: 色成分ベクトル。
        max_points: 表示する最大点数。
    """
    # SのRGB成分を平坦化して取得
    x = skin_flat[0].flatten()
    y = skin_flat[1].flatten()
    z = skin_flat[2].flatten()
    
    # サンプリング
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]

    # 点群を追加
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color='peachpuff', opacity=0.5)
    )])

    # 平面を追加
    melanin_vector = np.array(melanin_vector)
    hemoglobin_vector = np.array(hemoglobin_vector)
    u = np.linspace(0, 6, 30)
    v = np.linspace(0, 6, 30)
    U, V = np.meshgrid(u, v)
    plane_x = U * hemoglobin_vector[0] + V * melanin_vector[0]
    plane_y = U * hemoglobin_vector[1] + V * melanin_vector[1]
    plane_z = U * hemoglobin_vector[2] + V * melanin_vector[2]
    fig.add_trace(go.Surface(
        x=plane_x, y=plane_y, z=plane_z,
        colorscale='Blues', opacity=0.5
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='-logR Component',
            yaxis_title='-logG Component',
            zaxis_title='-logB Component'
        ),
        title="3D Plot of SkinFlat (Plotly)"
    )
    fig.show()