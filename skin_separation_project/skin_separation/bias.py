import numpy as np

def get_minvalue_after_outlier_removal(data, k, bins=30, title="Histogram after Adjustment"):
    """
    外れ値を除去した後の最小値を求める

    Parameters:
        data (array-like): 1次元データ。
        k (float): 許容範囲の倍率（デフォルト: 3σ）。
        bins (int): ヒストグラムのビンの数。
        title (str): グラフのタイトル。

    Returns:
        adjusted_data (array-like): 最小値が0に調整されたデータ。
    """
    # 平均と標準偏差を計算
    mean = np.mean(data)
    std = np.std(data)

    # 許容範囲を計算
    lower_bound = mean - k * std
    upper_bound = mean + k * std
    min_v = np.min(data)
    print(min_v)
    # 外れ値を除去
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # データの最小値を計算し、最小値が0になるように調整
    min_value = np.min(filtered_data)
    print(min_value)
    # # ヒストグラムの作成
    # fig = px.histogram(
    #     adjusted_data, nbins=bins,
    #     title=title,
    #     labels={'value': 'Value'},
    #     color_discrete_sequence=["blue"]
    # )

    # # グラフを表示
    # fig.show()

    return min_value

def find_optimal_offset(skin_flat, melanin, hemoglobin):
    """
    点群をメラニンとヘモグロビンベクトルの線形結合で表される範囲に多く含まれるよう移動する最適なオフセットを計算。

    Parameters:
        skin_flat: (3, height, width) のnumpy配列。点群。
        melanin: メラニンベクトル (3要素)。
        hemoglobin: ヘモグロビンベクトル (3要素)。

    Returns:
        optimal_offset: 点群を移動させる最適なベクトル (3要素)。
    """

    skin_flat_points = skin_flat.reshape(3, -1).T  # (N, 3)
    
    # メラニン・ヘモグロビンベクトルの基底を定義
    basis = np.array([melanin, hemoglobin]).T  # (3, 2)

    # 点が線形結合の範囲内に入るかをチェックする関数
    def is_within_range(points):
        """
        点群がメラニン・ヘモグロビンの正の線形結合で表されるか判定。
        """
        coefficients, residuals, rank, s = np.linalg.lstsq(basis, points.T, rcond=None)
        alpha, beta = coefficients
        return (alpha > 0) & (beta > 0)

    # 目的関数: 範囲内に入る点の割合を最大化
    def objective(offset):
        moved_points = skin_flat_points + offset  # 点群を移動
        within_range = is_within_range(moved_points)
        return -np.sum(within_range) + np.linalg.norm(offset)# 範囲内に入る点の数を最大化

    # 初期値（オフセットの初期推定）
    O0 = np.zeros(3)

    # 最適化実行
    result = minimize(objective, O0, method='Nelder-Mead')
    
    if not result.success:
        raise ValueError("Optimization failed.")
    
    optimal_offset = result.x
    return optimal_offset

#===============================================================
#バイアス処理実行部分
#===============================================================
#================
## 1次元の場合
#================

def bias_adjast_1d(L_Mel,L_Hem):
    L_Mel_flatten = L_Mel.flatten()
    L_Hem_flatten = L_Hem.flatten()
    
    min_Mel = get_minvalue_after_outlier_removal(L_Mel_flatten, k=3, bins=100, title="Histogram after Adjustment")
    min_Hem = get_minvalue_after_outlier_removal(L_Hem_flatten, k=3, bins=100, title="Histogram after Adjustment")
    print(f"{min_Mel=}")
    print(f"{min_Hem=}")
    
    if min_Mel <0:
        L_Mel = L_Mel - min_Mel
        
    if min_Hem < 0:
        L_Hem = L_Hem - min_Hem
            
    return L_Mel,L_Hem
