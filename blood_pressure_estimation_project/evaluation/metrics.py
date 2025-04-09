import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def calc_metrics(sbp_reference_array, dbp_reference_array, estimated_arrays,
                 selected_features_sbp, selected_features_dbp, ml_algorithm_feature_list):
    """
    各モデル・条件における血圧推定結果に対し、様々な評価指標を計算・出力する。
    評価項目：MAE, 相対誤差, Mean bias, SD, AAMI/BHS基準, 相関係数, 散布図保存

    Parameters
    ----------
    sbp_reference_array : ndarray
        収縮期血圧（実測値）
    dbp_reference_array : ndarray
        拡張期血圧（実測値）
    estimated_arrays : list of ndarray
        モデルによる予測値（SBP, DBP を交互に格納）
    selected_features_sbp : list
        選択されたSBP用特徴量のインデックス
    selected_features_dbp : list
        選択されたDBP用特徴量のインデックス
    ml_algorithm_feature_list : list of str
        使用したモデル・特徴量組み合わせの説明ラベル
    """

    mae_dict = {}
    results = []
    for i, ml_algorithm_feature in enumerate(ml_algorithm_feature_list):
        sbp_est = estimated_arrays[2 * i]
        dbp_est = estimated_arrays[2 * i + 1]

        # MAE
        mae_sbp = mean_absolute_error(sbp_reference_array, sbp_est)
        mae_dbp = mean_absolute_error(dbp_reference_array, dbp_est)
        # キー名は "sbp_モデル名_特徴種別"
        mae_dict[f"sbp_{ml_algorithm_feature}"] = mae_sbp
        mae_dict[f"dbp_{ml_algorithm_feature}"] = mae_dbp

        # 相対誤差
        relative_error_sbp = np.mean(np.abs(sbp_est - sbp_reference_array) / sbp_reference_array) * 100
        relative_error_dbp = np.mean(np.abs(dbp_est - dbp_reference_array) / dbp_reference_array) * 100

        # Mean Bias
        diff_sbp = sbp_reference_array - sbp_est
        diff_dbp = dbp_reference_array - dbp_est
        me_sbp = np.mean(diff_sbp)
        me_dbp = np.mean(diff_dbp)

        # SD
        sd_sbp = np.std(sbp_est)
        sd_dbp = np.std(dbp_est)

        # AAMI基準
        sbp_aami = determine_aami_standard(me_sbp, sd_sbp)
        dbp_aami = determine_aami_standard(me_dbp, sd_dbp)

        # BHS基準
        sbp_ratios = determine_bhs_grade(diff_sbp)
        dbp_ratios = determine_bhs_grade(diff_dbp)

        # 相関係数
        corr_sbp = np.corrcoef(sbp_reference_array, sbp_est)[0, 1]
        corr_dbp = np.corrcoef(dbp_reference_array, dbp_est)[0, 1]

        # 散布図保存
        fname_sbp = f"scatter_sbp_{ml_algorithm_feature.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fname_dbp = f"scatter_dbp_{ml_algorithm_feature.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_scatter(sbp_reference_array, sbp_est, fname_sbp)
        plot_scatter(dbp_reference_array, dbp_est, fname_dbp)

        results.append((ml_algorithm_feature, mae_sbp, mae_dbp, relative_error_sbp, relative_error_dbp,
                        me_sbp, me_dbp, sd_sbp, sd_dbp, sbp_aami, dbp_aami, sbp_ratios, dbp_ratios,
                        corr_sbp, corr_dbp))

    # 評価結果をファイルに保存
    with open("mae_results.txt", "a") as f:
        for r in results:
            f.write(f"{r[0]}\n")
            f.write(f"MAE SBP: {r[1]:.3f} mmHg | MAE DBP: {r[2]:.3f} mmHg\n")
            f.write(f"Relative Error SBP: {r[3]:.2f}% | DBP: {r[4]:.2f}%\n")
            f.write(f"Mean Bias SBP: {r[5]:.3f} mmHg | DBP: {r[6]:.3f} mmHg\n")
            f.write(f"SD SBP: {r[7]:.3f} mmHg | DBP: {r[8]:.3f} mmHg\n")
            f.write(f"AAMI SBP: {r[9]} | DBP: {r[10]}\n")
            f.write(f"BHS SBP (<5,10,15): {r[11][0]:.2f}, {r[11][1]:.2f}, {r[11][2]:.2f} => {r[11][3]}\n")
            f.write(f"BHS DBP (<5,10,15): {r[12][0]:.2f}, {r[12][1]:.2f}, {r[12][2]:.2f} => {r[12][3]}\n")
            f.write(f"Correlation SBP: {r[13]:.3f} | DBP: {r[14]:.3f}\n\n")

        if selected_features_sbp:
            f.write(f"Selected Features for SBP: {selected_features_sbp}\n")
        if selected_features_dbp:
            f.write(f"Selected Features for DBP: {selected_features_dbp}\n")
        f.write("\n")

    return mae_dict


def determine_aami_standard(me, sd):
    """
    AAMI基準に基づいて、mean bias (ME)と標準偏差 (SD) を評価
    """
    return abs(me) <= 5 and sd <= 8


def determine_bhs_grade(diff_array):
    """
    BHS基準に基づいて、<5, <10, <15mmHgの割合と等級を算出
    """
    abs_diff = np.abs(diff_array)
    r5 = np.sum(abs_diff < 5) / len(abs_diff)
    r10 = np.sum(abs_diff < 10) / len(abs_diff)
    r15 = np.sum(abs_diff < 15) / len(abs_diff)

    if r5 >= 0.6 and r10 >= 0.85 and r15 >= 0.95:
        grade = "A"
    elif r5 >= 0.5 and r10 >= 0.75 and r15 >= 0.90:
        grade = "B"
    elif r5 >= 0.4 and r10 >= 0.60 and r15 >= 0.85:
        grade = "C"
    else:
        grade = "D"
    return r5, r10, r15, grade


def plot_scatter(reference, estimated, filename):
    """
    散布図を生成して保存
    """
    plt.figure()
    plt.scatter(reference, estimated, alpha=0.6)
    plt.xlabel("Measured BP")
    plt.ylabel("Estimated BP")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()