import os
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.stats import zscore
import yaml

def load_subject_ages(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("subject_ages", {})



def load_features_with_age(base_dir, feature_dir, state_list, feature_num_cn, feature_num_dr, age_dict=None):
    """
    特徴量を読み込んで、G成分（cn+dr）の特徴量 + 年齢の配列にして返す
    """
    subject_dirs = [d for d in natsorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    num_subjects = len(subject_dirs)
    num_states = len(state_list)
    feature_num_all = feature_num_cn + feature_num_dr

    # 出力配列（最後の+1は年齢特徴量）
    arr_rgb_g = np.empty((num_subjects, num_states, feature_num_all + 1))

    for i, subject in enumerate(subject_dirs):
        subject_path = os.path.join(base_dir, subject)

        for j, state in enumerate(state_list):
            base_state_dir = os.path.join(subject_path, subject + state, feature_dir)
            try:
                f_cn_g = np.loadtxt(os.path.join(base_state_dir, "features_cn.csv"), delimiter=",")
                f_dr_g = np.loadtxt(os.path.join(base_state_dir, "features_dr.csv"), delimiter=",")
            except FileNotFoundError:
                print(f"[SKIP] {subject} - {state}: G成分の特徴量が見つかりません")
                continue

            # cn + dr のG成分特徴量を結合
            f_g = np.concatenate([f_cn_g, f_dr_g])

            # 年齢情報を末尾に追加
            age = age_dict.get(subject, 25) if age_dict else 25
            f_g = np.append(f_g, age)

            # 標準化（Zスコア）
            f_g = zscore(f_g)

            # 配列に格納
            arr_rgb_g[i, j] = f_g

    return arr_rgb_g, subject_dirs


def load_all_data(base_dir, feature_dir, bp_dir, age_yaml_path, feature_num_cn, feature_num_dr):
    # 年齢読み込み
    with open(age_yaml_path, "r") as f:
        age_dict = yaml.safe_load(f)["subject_ages"]

    data = []  # 各行をここに追加

    subject_dirs = [d for d in natsorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]

    for subject in subject_dirs:
        subject_path = os.path.join(base_dir, subject)
        print(subject_path)

        # 血圧ファイル読み込み（例：subject1.csv）
        bp_path = os.path.join(bp_dir, f"{subject}.csv")
        # print(bp_path)
        try:
            bp_df = pd.read_csv(bp_path)  # ← SBP, DBP列で読み込める
            bp_values = bp_df[["SBP", "DBP"]].values
        except:
            print(f"[SKIP] 血圧ファイルが読み込めない: {bp_path}")
            continue

        age = age_dict.get(subject, 25)

        # 試行ディレクトリを探索（例：subject1_ex_01）
        trial_dirs = [
        d for d in natsorted(os.listdir(subject_path))
        if d.startswith(subject + "_ex_") and os.path.isdir(os.path.join(subject_path, d))]
    
        for idx, trial in enumerate(trial_dirs):
        
            trial_path = os.path.join(subject_path,trial)
            name, ext = os.path.splitext(trial_path)
            trial_path = os.path.join(name,feature_dir)
            try:
                f_cn = np.loadtxt(os.path.join(trial_path, "features_cn.csv"), delimiter=",")
                f_dr = np.loadtxt(os.path.join(trial_path, "features_dr.csv"), delimiter=",")
            except FileNotFoundError:
                print(f"[SKIP] 特徴量ファイルが見つかりません: {trial}")
                continue

            features = np.concatenate([f_cn, f_dr])
            features = zscore(features)
            # 血圧値を取得（例：1行ずつ）
            try:
                sbp, dbp = bp_values[idx]
            except:
                print(f"[SKIP] 血圧データ不足: {subject}, {trial}")
                sbp, dbp = np.nan, np.nan

            row = {
                "subject": subject,
                "trial": trial,
                "age": age,
                "SBP": sbp,
                "DBP": dbp,
            }

            # 特徴量をfeature_1, feature_2...の形で追加
            for i, val in enumerate(features):
                row[f"feature_{i+1}"] = val

            data.append(row)

    df = pd.DataFrame(data)
    return df


def df_to_np_arrays(df, feature_prefix="feature_", num_trials_per_subject=6):
    # 被験者ごとのデータをグループ化
    subjects = df["subject"].unique()
    num_subjects = len(subjects)
    
    # 特徴量の列を抽出
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    num_features = len(feature_cols)

    # 初期化
    feature_array = np.zeros((num_subjects, num_trials_per_subject, num_features))
    bp_array = np.zeros((num_subjects * num_trials_per_subject, 2))  # SBP, DBP

    for i, subject in enumerate(subjects):
        subject_df = df[df["subject"] == subject]
        subject_df = subject_df.sort_values("trial")  # 自然順が必要なら natsorted を検討

        # 特徴量
        feature_array[i] = subject_df[feature_cols].values

        # 血圧
        bp_array[i * num_trials_per_subject : (i + 1) * num_trials_per_subject] = subject_df[["SBP", "DBP"]].values

    return feature_array, bp_array