import matplotlib.pyplot as plt
import numpy as np
import os

def plot_scatter(bp_reference, bp_estimated, filename):
    """
    実測血圧と推定血圧の散布図を描画してファイルに保存する。

    Parameters
    ----------
    bp_reference : array-like
        実測値（正解データ）
    bp_estimated : array-like
        推定値（予測結果）
    filename : str
        保存する画像ファイル名（.png）
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(bp_reference, bp_estimated, c='blue', alpha=0.6, edgecolors='k')
    plt.xlabel("Measured BP [mmHg]")
    plt.ylabel("Estimated BP [mmHg]")
    plt.title("Estimated vs Measured BP")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"[Saved] {filename}")