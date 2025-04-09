# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 10:05:59 2024

@author: pinyo
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from natsort import natsorted
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

"""
機械学習手法→パラメータチューニング→特徴量選択→評価手法

20241126 ages追加

"""
def save_subject_results_to_csv(subject_results, camera_type):
    """
    被験者ごとの推定結果をCSVファイルに保存する

    Parameters
    ----------
    subject_results : list
        各被験者の結果を格納したリスト
    camera_type : str
        カメラの種類（例: "RGB-NIR G"）
    """
    output_dir = "subject_results\\"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{camera_type}_subject_results.csv")
    
    # データフレームに変換して保存
    df = pd.DataFrame(subject_results)
    df.to_csv(output_file, index=False)
    print(f"Subject results saved to {output_file}")

def save_predictions_to_txt(predictions):
    OUTPUT_FILE = "predictions.txt"
    with open(OUTPUT_FILE, "a") as file:
        for key, values in predictions.items():
            file.write(f"{key}:\n")
            file.write(", ".join(map(str, values)) + "\n")
            file.write("\n")

def save_best_params(camera_type, best_params):
    OUTPUT_FILE = "best_params.txt"
    with open(OUTPUT_FILE, "a") as f:  # 追記モードで開く
        f.write(f"Camera Type: {camera_type}\n")
        for model in best_params:
            f.write(f"{model}:\n")
            for param in best_params[model]:
                f.write(f"  {param}: {best_params[model][param]}\n")
            f.write("\n")

def save_selected_features(camera_type, model_name, target, selected_features):
    OUTPUT_FILE = "selected_features.txt"
    with open(OUTPUT_FILE, "a") as f:  # ファイルに追記
        f.write(f"Camera Type: {camera_type}, Model: {model_name}, Target: {target}\n")
        f.write(f"Selected Features: {selected_features}\n\n")

def save_model(model, model_name, target, select, camera_type):
    MODEL_SAVE_DIR = "saved_models\\"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    file_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{target}_{select}_{camera_type}.pkl")
    joblib.dump(model, file_path)

def objective_rf(trial, X, y):
    
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                  max_features=max_features)
    
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae = -score.mean()
    return mae

def objective_svr(trial, X, y):
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    epsilon = trial.suggest_loguniform("epsilon", 1e-3, 1e1)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
    else:
        degree = 3  # Default value
    
    model = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree)
    
    # クロスバリデーションでの評価
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae = -score.mean()
    return mae

def objective_lgbm(trial, X, y):
    num_leaves = trial.suggest_int("num_leaves", 2, 256)
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1.0)
    feature_fraction = trial.suggest_uniform("feature_fraction", 0.4, 1.0)
    bagging_fraction = trial.suggest_uniform("bagging_fraction", 0.4, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
    
    model = LGBMRegressor(num_leaves=num_leaves, n_estimators=n_estimators,
                          learning_rate=learning_rate, feature_fraction=feature_fraction,
                          bagging_fraction=bagging_fraction, bagging_freq=bagging_freq)
    
    # クロスバリデーションでの評価
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae = -score.mean()
    return mae

def estimate_bp(feature_array, bp_array, num_experiments, camera_type, num_state_count):
    """
    血圧推定を行う．

    Parameters
    ----------
    feature_array: np.ndarray
        特徴量が格納されたNumPy配列
    num_experiments: int
        被験者数
    """
    # Nanがある場合はその列の平均値で補完
    for subject_idx, features in enumerate(feature_array):  # feature_array の配列を確認
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
    
        if nan_count > 0 or inf_count > 0:
            print(f"被験者 {subject_idx} のデータに NaN または Infinity が含まれています")
            
            # 列ごとの平均値を計算 (NaNを無視して計算)
            mean_vals = np.nanmean(features, axis=0)
            
            # NaNを列ごとの平均値で埋める
            features = np.where(np.isnan(features), mean_vals, features)
            
            # Infinityを最大値または最小値で埋める
            max_val = np.nanmax(features[np.isfinite(features)])  # 無限大以外の最大値
            features = np.where(np.isinf(features), max_val, features)
            
            # データ型を確認
            if features.dtype != np.float32:
                features = features.astype(np.float32)
                
        nan_count_final = np.isnan(features).sum()
        inf_count_final = np.isinf(features).sum()
        
        if nan_count_final != 0 or inf_count_final != 0:
            print(f"NaN count after cleaning: {nan_count_final}")
            print(f"Infinity count after cleaning: {inf_count_final}")

        # 修正した特徴量を元の配列に戻す
        feature_array[subject_idx] = features
            
    
    # 収縮期血圧 (Systolic blood pressure, SBP) と拡張期血圧 (Diastolic blood pressure, DBP) の正解値を格納するリストを用意
    sbp_reference_arr = []
    dbp_reference_arr = []

    # SBPとDBPの予測値を格納するリストを用意
    # all: 全特徴量, rfe: RFE(recursive feature elimination)で選択された特徴量
    sbp_predicted_rf_all_arr = []
    dbp_predicted_rf_all_arr = []
    sbp_predicted_rf_rfe_arr = []
    dbp_predicted_rf_rfe_arr = []
    sbp_predicted_svr_all_arr = []
    dbp_predicted_svr_all_arr = []
    sbp_predicted_svr_rfe_arr = []
    dbp_predicted_svr_rfe_arr = []
    sbp_predicted_lgbm_all_arr = []
    dbp_predicted_lgbm_all_arr = []
    sbp_predicted_lgbm_rfe_arr = []
    dbp_predicted_lgbm_rfe_arr = []


    # 特徴量カウント用リスト
    selected_features_sbp_count_list = [0] * len(feature_array[0][0])
    selected_features_dbp_count_list = [0] * len(feature_array[0][0])
    
    
    # 機械学習アルゴリズムを用いて血圧推定を行うため，訓練データとテストデータに分ける
    
    """
    #LOSO
    for test_idx in range(num_experiments):
        idx_list = np.arange(num_experiments)
        # idx_listからtest_idx（テストデータ用のインデックス）の値を削除し，残りを訓練データ用のインデックスとする
        train_idx_list = idx_list[idx_list != test_idx]

        X_train = np.concatenate([feature_array[i] for i in train_idx_list])
        X_test = feature_array[test_idx]

        y_train = np.concatenate([bp_array[6 * i: 6 * i + 6] for i in train_idx_list])
        y_test = bp_array[6 * test_idx: 6 * test_idx + 6]

        # y_trainを収縮期血圧 (Systolic blood pressure, SBP) と拡張期血圧 (Diastolic blood pressure, DBP) に分ける
        y_train_sbp = y_train[:, 0]
        y_train_dbp = y_train[:, 1]
        y_test_sbp = y_test[:, 0]
        y_test_dbp = y_test[:, 1]
    """
    """
    # 普通のkfold
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(np.arange(num_experiments)):
        # 各被験者のデータを訓練データとテストデータに分ける
        X_train = np.concatenate([feature_array[i] for i in train_index])
        X_test = np.concatenate([feature_array[i] for i in test_index])

        y_train = np.concatenate([bp_array[6 * i: 6 * i + 6] for i in train_index])
        y_test = np.concatenate([bp_array[6 * i: 6 * i + 6] for i in test_index])

        # y_trainを収縮期血圧 (Systolic blood pressure, SBP) と拡張期血圧 (Diastolic blood pressure, DBP) に分ける
        y_train_sbp = y_train[:, 0]
        y_train_dbp = y_train[:, 1]
        y_test_sbp = y_test[:, 0]
        y_test_dbp = y_test[:, 1]
    """
    """
    # 必ず全員のデータがtrain, testに入るようにする
    for fold_idx in range(6):  # 状態の数（6状態分）
        test_state = fold_idx  # テストデータとして使用する状態
        train_states = [i for i in range(6) if i != fold_idx]  # 残りの状態を訓練データに
    
        # 各状態をtrainとtestに分ける（同じ被験者のデータが含まれるようにする）
        X_train = np.concatenate([feature_array[:, state, :] for state in train_states], axis=0)
        X_test = feature_array[:, test_state, :]
    
        y_train = np.concatenate([bp_array[state::6] for state in train_states], axis=0)
        y_test = bp_array[test_state::6]
    
        # y_trainを収縮期血圧 (SBP) と拡張期血圧 (DBP) に分ける
        y_train_sbp = y_train[:, 0]
        y_train_dbp = y_train[:, 1]
        y_test_sbp = y_test[:, 0]
        y_test_dbp = y_test[:, 1]
    """
    subject_results = [] 
    # フラット化
    flat_feature_array = []
    subject_indices = []
    
    print(len(feature_array))
    
    for subject_idx, subject_data in enumerate(feature_array):
        print(f"Subject {subject_idx}: {len(subject_data)} states")
        flat_feature_array.extend(subject_data)
        subject_indices.extend([subject_idx] * len(subject_data))
    
    print(f"Total states: {sum(len(subject) for subject in feature_array)}")
    cumulative_indices = np.cumsum([0] + [len(subject) for subject in feature_array])
    

    #LOSO
    for test_idx in range(num_experiments):
        #idx_list = np.arange(num_experiments)
        # idx_listからtest_idx（テストデータ用のインデックス）の値を削除し，残りを訓練データ用のインデックスとする
        #train_idx_list = idx_list[idx_list != test_idx]

        # テストデータの範囲
        test_start = cumulative_indices[test_idx]
        test_end = cumulative_indices[test_idx + 1]
     
        # 訓練データのインデックスを取得（テスト範囲を除く）
        train_indices = np.arange(cumulative_indices[-1])
        train_indices = np.delete(train_indices, np.arange(test_start, test_end))
     
        # 特徴量とラベルを取得
        X_train = np.vstack([flat_feature_array[i] for i in train_indices])  # ジェネレータをリストに変換
        X_test = np.vstack(flat_feature_array[test_start:test_end])  # テストデータを2次元配列で取得
        
        
        y_train = np.concatenate([bp_array[i][np.newaxis, :] for i in train_indices], axis=0)
        y_test = bp_array[test_start:test_end, :]  # 明示的に2次元スライス


        # y_trainを収縮期血圧 (Systolic blood pressure, SBP) と拡張期血圧 (Diastolic blood pressure, DBP) に分ける
        y_train_sbp = y_train[:, 0]
        y_train_dbp = y_train[:, 1]
        y_test_sbp = y_test[:, 0]
        y_test_dbp = y_test[:, 1]
        

        # 機械学習アルゴリズムを用いて血圧推定を行い，予測値を取得 
        sbp_predicted_rf_all, dbp_predicted_rf_all, sbp_predicted_rf_rfe, dbp_predicted_rf_rfe, \
        sbp_predicted_svr_all, dbp_predicted_svr_all, sbp_predicted_svr_rfe, dbp_predicted_svr_rfe, \
        sbp_predicted_lgbm_all, dbp_predicted_lgbm_all, sbp_predicted_lgbm_rfe, dbp_predicted_lgbm_rfe, \
        selected_feature_indices_sbp, selected_feature_indices_dbp = perform_machine_learning(X_train, X_test, y_train_sbp, y_train_dbp, camera_type)
        
        
        # 被験者ごとの結果を記録
        subject_results.append({
            "subject_id": test_idx,
            "sbp_true": list(y_test_sbp),
            "sbp_predicted_rf_all": list(sbp_predicted_rf_all),
            "sbp_predicted_svr_all": list(sbp_predicted_svr_all),
            "sbp_predicted_lgbm_all": list(sbp_predicted_lgbm_all),
            "dbp_true": list(y_test_dbp),
            "dbp_predicted_rf_all": list(dbp_predicted_rf_all),
            "dbp_predicted_svr_all": list(dbp_predicted_svr_all),
            "dbp_predicted_lgbm_all": list(dbp_predicted_lgbm_all),
        })
        
        
        save_subject_results_to_csv(subject_results, camera_type)
        
        # SBP, DBPの正解値をリストに格納
        sbp_reference_arr.extend(y_test_sbp)
        dbp_reference_arr.extend(y_test_dbp)

        # SBP, DBPの予測値をリストに格納
        sbp_predicted_rf_all_arr.extend(sbp_predicted_rf_all)
        dbp_predicted_rf_all_arr.extend(dbp_predicted_rf_all)
        sbp_predicted_rf_rfe_arr.extend(sbp_predicted_rf_rfe)
        dbp_predicted_rf_rfe_arr.extend(dbp_predicted_rf_rfe)
        sbp_predicted_svr_all_arr.extend(sbp_predicted_svr_all)
        dbp_predicted_svr_all_arr.extend(dbp_predicted_svr_all)
        sbp_predicted_svr_rfe_arr.extend(sbp_predicted_svr_rfe)
        dbp_predicted_svr_rfe_arr.extend(dbp_predicted_svr_rfe)
        sbp_predicted_lgbm_all_arr.extend(sbp_predicted_lgbm_all)
        dbp_predicted_lgbm_all_arr.extend(dbp_predicted_lgbm_all)
        sbp_predicted_lgbm_rfe_arr.extend(sbp_predicted_lgbm_rfe)
        dbp_predicted_lgbm_rfe_arr.extend(dbp_predicted_lgbm_rfe)


        # 特徴量カウント用リストの更新
        # selected_features_indices_sbp/dbp内の要素について，selected_features_..._count_list内のインデックスの要素を+1する
        selected_features_sbp_count_list = [count + 1 if index in selected_feature_indices_sbp else count for index, count in enumerate(selected_features_sbp_count_list)]
        selected_features_dbp_count_list = [count + 1 if index in selected_feature_indices_dbp else count for index, count in enumerate(selected_features_dbp_count_list)]
        

    # selected_features_..._count_listを降順にソートし，対応するインデックスを出力 (=各特徴量の選択数が降順に格納される)
    sorted_indices_sbp = sorted(range(len(selected_features_sbp_count_list)), key=lambda k: selected_features_sbp_count_list[k], reverse=True)
    sorted_indices_dbp = sorted(range(len(selected_features_dbp_count_list)), key=lambda k: selected_features_dbp_count_list[k], reverse=True)
    print()
    print("Features of SBP:", sorted_indices_sbp[:25])
    print("Features of DBP:", sorted_indices_dbp[:25])
    print()
    sbp_sorted_list = sorted(selected_features_sbp_count_list, reverse=True)
    dbp_sorted_list = sorted(selected_features_dbp_count_list, reverse=True)
    print(sbp_sorted_list)
    print(dbp_sorted_list)
    print()
    
    
    # 予測値をリストにまとめる (今後の計算のため，リストからNumPy配列への変換を行う)
    estimated_arr = [np.array(sbp_predicted_rf_all_arr), np.array(dbp_predicted_rf_all_arr),
                     np.array(sbp_predicted_rf_rfe_arr), np.array(dbp_predicted_rf_rfe_arr),
                     np.array(sbp_predicted_svr_all_arr), np.array(dbp_predicted_svr_all_arr),
                     np.array(sbp_predicted_svr_rfe_arr), np.array(dbp_predicted_svr_rfe_arr),
                     np.array(sbp_predicted_lgbm_all_arr), np.array(dbp_predicted_lgbm_all_arr),
                     np.array(sbp_predicted_lgbm_rfe_arr), np.array(dbp_predicted_lgbm_rfe_arr)]
    
    # 評価指標の計算 (main関数の最後でMAEを一括して表示するため，mae_listをmain関数に返すように設定)
    sbp_reference_arr = np.array(sbp_reference_arr)
    dbp_reference_arr = np.array(dbp_reference_arr)
    mae_list = calc_metrics(sbp_reference_arr, dbp_reference_arr, estimated_arr, camera_type, selected_feature_indices_sbp, selected_feature_indices_dbp)
    
    return mae_list, np.array(sbp_predicted_svr_all_arr), np.array(dbp_predicted_svr_all_arr)

def get_model(model_name, params, rand):
    if model_name == "rf":
        return RandomForestRegressor(**params)
    elif model_name == "svr":
        return SVR(**params)
    elif model_name == "lgbm":
        return LGBMRegressor(**params)
    elif model_name == "gbr":
        return GradientBoostingRegressor(**params)
    elif model_name == "ridge":
        return Ridge(**params)
    else:
        raise ValueError("Invalid model name")


def perform_machine_learning(X_train, X_test, y_train_sbp, y_train_dbp, camera_type, rand=0):
    """
    血圧推定を行うため，機械学習アルゴリズムを用いる．

    Parameters
    ----------
    X_train: np.ndarray
        訓練データ (特徴量)
    X_test: np.ndarray
        テストデータ (特徴量)
    y_train_sbp: np.ndarray
        訓練データ (血圧計で計測した収縮期血圧)
    y_train_dbp: np.ndarray
        訓練データ（血圧計で計測した拡張期血圧）
    rand: int
        random_stateを固定する値 (適当な値)
    """
    #パラメータチューニング
    param_flag = True
    
    studies = {}
    best_params = {}

    
    # 初期化
    default_params = {
       "rf_sbp": {"n_estimators": 100, "random_state": rand},
       "rf_dbp": {"n_estimators": 100, "random_state": rand},
       "svr_sbp": {"kernel": "rbf", "gamma": 20},
       "svr_dbp": {"kernel": "rbf", "gamma": 20},
       "lgbm_sbp": {"random_state": rand},
       "lgbm_dbp": {"random_state": rand},
       "gbr_sbp": { "random_state": rand},
       "gbr_dbp": { "random_state": rand},
       "ridge_sbp": { "random_state": rand},
       "ridge_dbp": { "random_state": rand}
    }
    
    if param_flag == True:
        
        for target, y_train in zip(["sbp", "dbp"], [y_train_sbp, y_train_dbp]):
            for model_name, objective_func in zip(
                    #["rf", "svr", "lgbm", "gbr", "ridge"],
                    ["rf", "svr", "lgbm"],
                    #[objective_rf, objective_svr, objective_lgbm, objective_gbr, objective_ridge]):
                    [objective_rf, objective_svr, objective_lgbm]):
                try:
                    # Optunaでハイパーパラメータチューニング
                    study = optuna.create_study(direction="minimize")
                    study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=100)
                    studies[f"{model_name}_{target}"] = study
                    optuna_best_params = study.best_params
                except ValueError as e:
                    # Optunaが失敗した場合
                    print(f"Optuna failed for {model_name}_{target}. Using default parameters. Error: {e}")
                    optuna_best_params = default_params[f"{model_name}_{target}"]
        
                # デフォルトパラメータのモデルと比較
                model_optuna = get_model(model_name, optuna_best_params, rand)
                model_default = get_model(model_name, default_params[f"{model_name}_{target}"], rand)
    
                # トレーニングと予測
                model_optuna.fit(X_train, y_train)
                model_default.fit(X_train, y_train)
    
                # クロスバリデーションによる評価
                scores_optuna = cross_val_score(model_optuna, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
                scores_default = cross_val_score(model_default, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    
                mse_optuna = -scores_optuna.mean()
                mse_default = -scores_default.mean()
    
    
                # 良い方をbest_paramsに追加
                if mse_optuna < mse_default:
                    best_params[f"{model_name}_{target}"] = optuna_best_params
                else:
                    best_params[f"{model_name}_{target}"] = default_params[f"{model_name}_{target}"]
            
    else:
        best_params = default_params
        
    # 最適パラメータをテキストファイルに出力
    save_best_params(camera_type, best_params)
    

    #n_features_to_select = 25

    #特徴量の個数は自動選択
    
    # # y_trainを収縮期血圧 (Systolic blood pressure, SBP) と拡張期血圧 (Diastolic blood pressure, DBP) に分ける
    # y_train_sbp = y_train[:, 0]
    # y_train_dbp = y_train[:, 1]

    # ランダムフォレスト (random forest, rf)-------------------------------
    # SBP (全特徴量)
    rf_sbp_regressor = RandomForestRegressor(**best_params["rf_sbp"])
    rf_sbp_regressor.fit(X_train, y_train_sbp)
    rf_sbp_predicted = rf_sbp_regressor.predict(X_test)
    save_model(rf_sbp_regressor, "rf", "sbp", "all", camera_type)

    # DBP (全特徴量)
    rf_dbp_regressor = RandomForestRegressor(**best_params["rf_dbp"])
    rf_dbp_regressor.fit(X_train, y_train_dbp)
    rf_dbp_predicted = rf_dbp_regressor.predict(X_test)
    save_model(rf_dbp_regressor, "rf", "dbp", "all", camera_type)

    # SBP (RFE (recursive feature elimination) によって選択された特徴量のみ)
    rf_rfe_sbp = RandomForestRegressor(**best_params["rf_sbp"])
    #rfe_sbp = RFE(estimator=rf_rfe_sbp, n_features_to_select=n_features_to_select, verbose=0)
    rfe_sbp = RFECV(estimator=rf_rfe_sbp, step=1, cv=5, scoring='neg_mean_absolute_error', verbose=0)
    X_train_rfe_sbp = rfe_sbp.fit_transform(X_train, y_train_sbp)
    X_test_rfe_sbp = rfe_sbp.transform(X_test)
    rf_rfe_sbp.fit(X_train_rfe_sbp, y_train_sbp)
    rf_rfe_sbp_predicted = rf_rfe_sbp.predict(X_test_rfe_sbp)
    # 選択された特徴量のインデックス
    selected_feature_indices_sbp = [i for i in range(len(rfe_sbp.support_)) if rfe_sbp.support_[i]]
    # print("Features of SBP (RFE):")
    # print(selected_feature_indices_sbp)
    save_model(rf_rfe_sbp, "rf", "sbp", "rfe", camera_type)
    save_selected_features(camera_type, "RandomForest", "SBP", selected_feature_indices_sbp)

    # DBP (RFE後)
    rf_rfe_dbp = RandomForestRegressor(**best_params["rf_dbp"])
    #rfe_dbp = RFE(estimator=rf_rfe_dbp, n_features_to_select=n_features_to_select, verbose=0)
    rfe_dbp = RFECV(estimator=rf_rfe_dbp, step=1, cv=5, scoring='neg_mean_absolute_error', verbose=0)
    X_train_rfe_dbp = rfe_dbp.fit_transform(X_train, y_train_dbp)
    X_test_rfe_dbp = rfe_dbp.transform(X_test)
    rf_rfe_dbp.fit(X_train_rfe_dbp, y_train_dbp)
    rf_rfe_dbp_predicted = rf_rfe_dbp.predict(X_test_rfe_dbp)
    # 選択された特徴量のインデックス
    selected_feature_indices_dbp = [i for i in range(len(rfe_dbp.support_)) if rfe_dbp.support_[i]]
    # print("Features of DBP (RFE):")
    # print(selected_feature_indices_dbp)
    save_model(rf_rfe_dbp, "rf", "dbp", "rfe", camera_type)
    save_selected_features(camera_type, "RandomForest", "DBP", selected_feature_indices_dbp)
    

    # support vector regression (SVR)--------------------------------------
    # SBP (全特徴量)
    # svr_sbp_regressor = SVR(kernel='linear')
    svr_sbp_regressor = SVR(**best_params["svr_sbp"])
    svr_sbp_regressor.fit(X_train, y_train_sbp)
    svr_sbp_predicted = svr_sbp_regressor.predict(X_test)
    save_model(svr_sbp_regressor, "svr", "sbp", "all", camera_type)

    # DBP (全特徴量)
    # svr_dbp_regressor = SVR(kernel='linear')
    svr_dbp_regressor = SVR(**best_params["svr_dbp"])
    svr_dbp_regressor.fit(X_train, y_train_dbp)
    svr_dbp_predicted = svr_dbp_regressor.predict(X_test)
    save_model(svr_dbp_regressor, "svr", "dbp", "all", camera_type)

    # SBP (RFE後)
    svr_sbp_rfe = SVR(**best_params["svr_sbp"])
    svr_sbp_rfe.fit(X_train_rfe_sbp, y_train_sbp)
    svr_sbp_rfe_predicted = svr_sbp_rfe.predict(X_test_rfe_sbp)
    save_model(svr_sbp_rfe, "svr", "sbp", "rfe", camera_type)
    save_selected_features(camera_type, "svr", "SBP", selected_feature_indices_sbp)

    # DBP (RFE後)
    svr_dbp_rfe = SVR(**best_params["svr_dbp"])
    svr_dbp_rfe.fit(X_train_rfe_dbp, y_train_dbp)
    svr_dbp_rfe_predicted = svr_dbp_rfe.predict(X_test_rfe_dbp)
    save_model(svr_dbp_regressor, "svr", "dbp", "rfe", camera_type)
    save_selected_features(camera_type, "svr", "SBP", selected_feature_indices_sbp)
    
    # LightGBM---------------------------------------------------------
    # SBP (全特徴量)
    lgbm_sbp_regressor = LGBMRegressor(**best_params["lgbm_sbp"])
    lgbm_sbp_regressor.fit(X_train, y_train_sbp)
    lgbm_sbp_predicted = lgbm_sbp_regressor.predict(X_test)
    save_model(lgbm_sbp_regressor, "lgbm", "sbp", "all", camera_type)

    # DBP (全特徴量)
    lgbm_dbp_regressor = LGBMRegressor(**best_params["lgbm_dbp"])
    lgbm_dbp_regressor.fit(X_train, y_train_dbp)
    lgbm_dbp_predicted = lgbm_dbp_regressor.predict(X_test)
    save_model(lgbm_dbp_regressor, "lgbm", "dbp", "all", camera_type)

    # SBP (RFE後)
    lgbm_rfe_sbp = LGBMRegressor(**best_params["lgbm_sbp"])
    #rfe_sbp_lgbm = RFE(estimator=lgbm_rfe_sbp, n_features_to_select=n_features_to_select, verbose=0)
    rfe_sbp_lgbm = RFECV(estimator=lgbm_rfe_sbp, step=1, cv=5, scoring='neg_mean_absolute_error', verbose=0)
    X_train_rfe_sbp_lgbm = rfe_sbp_lgbm.fit_transform(X_train, y_train_sbp)
    X_test_rfe_sbp_lgbm = rfe_sbp_lgbm.transform(X_test)
    lgbm_rfe_sbp.fit(X_train_rfe_sbp_lgbm, y_train_sbp)
    lgbm_rfe_sbp_predicted = lgbm_rfe_sbp.predict(X_test_rfe_sbp_lgbm)
    
    selected_feature_indices_sbp_lgbm = [i for i in range(len(rfe_sbp_lgbm.support_)) if rfe_sbp_lgbm.support_[i]]
    save_selected_features(camera_type, "LightGBM", "SBP", selected_feature_indices_sbp_lgbm)
    
    save_model(lgbm_rfe_sbp, "lgbm", "sbp", "rfe", camera_type)

    # DBP (RFE後)
    lgbm_rfe_dbp = LGBMRegressor(**best_params["lgbm_dbp"])
    #rfe_dbp_lgbm = RFE(estimator=lgbm_rfe_dbp, n_features_to_select=n_features_to_select, verbose=0)
    rfe_dbp_lgbm = RFECV(estimator=lgbm_rfe_dbp, step=1, cv=5, scoring='neg_mean_absolute_error', verbose=0)
    X_train_rfe_dbp_lgbm = rfe_dbp_lgbm.fit_transform(X_train, y_train_dbp)
    X_test_rfe_dbp_lgbm = rfe_dbp_lgbm.transform(X_test)
    lgbm_rfe_dbp.fit(X_train_rfe_dbp_lgbm, y_train_dbp)
    lgbm_rfe_dbp_predicted = lgbm_rfe_dbp.predict(X_test_rfe_dbp_lgbm)
    
    selected_feature_indices_dbp_lgbm = [i for i in range(len(rfe_dbp_lgbm.support_)) if rfe_dbp_lgbm.support_[i]]
    save_selected_features(camera_type, "LightGBM", "DBP", selected_feature_indices_dbp_lgbm)
    
    save_model(lgbm_rfe_dbp, "lgbm", "dbp", "rfe", camera_type)
    
    
    return rf_sbp_predicted, rf_dbp_predicted, rf_rfe_sbp_predicted, rf_rfe_dbp_predicted, \
           svr_sbp_predicted, svr_dbp_predicted, svr_sbp_rfe_predicted, svr_dbp_rfe_predicted, \
           lgbm_sbp_predicted, lgbm_dbp_predicted, lgbm_rfe_sbp_predicted, lgbm_rfe_dbp_predicted, \
           selected_feature_indices_sbp, selected_feature_indices_dbp


def calc_metrics(sbp_reference_array, dbp_reference_array, estimated_arrays, camera_type, selected_features_sbp, selected_features_dbp):
    """
    評価指標の計算（平均絶対誤差，Mean bias，標準偏差，AAMI基準，BHS基準）を行う．

    Parameters
    ----------
    sbp_reference_array: np.ndarray
        血圧計で測定したSBP (収縮期血圧)
    dbp_reference_array: np.ndarray
        血圧計で測定したDBP (拡張期血圧)
    estimated_arrays: list
        Random ForestやSVRで推定したSBPやDBPが格納されているNumPy配列をまとめたリスト

    Returns
    -------

    """
    mae_list = []
    results = []
    """
    ml_algorithm_feature_list = ["Random Forest (all features)", "Random Forest (selected features)",
                                 "SVR (all features)", "SVR (selected features)",
                                 "LightGBM (all features)", "LightGBM (selected features)",
                                 "Gradient Boosting (all features)", "Gradient Boosting (selected features)",
                                 "Logistic Regression (all features)", "Logistic Regression (selected features)"]
    """
    
    ml_algorithm_feature_list = ["Random Forest (all features)", "Random Forest (selected features)",
                                 "SVR (all features)", "SVR (selected features)",
                                 "LightGBM (all features)", "LightGBM (selected features)"]

    for i, ml_algorithm_feature in enumerate(ml_algorithm_feature_list):
        sbp_estimated_array = estimated_arrays[2 * i]
        dbp_estimated_array = estimated_arrays[2 * i + 1]

        # MAE
        mae_sbp = mean_absolute_error(sbp_reference_array, sbp_estimated_array)
        mae_dbp = mean_absolute_error(dbp_reference_array, dbp_estimated_array)

        print("MAE of SBP ({}): {:.3f} [mmHg]".format(ml_algorithm_feature, mae_sbp))
        print("MAE of DBP ({}): {:.3f} [mmHg]".format(ml_algorithm_feature, mae_dbp))

        mae_list.append(mae_sbp)
        mae_list.append(mae_dbp)
        
        relative_error_sbp = np.mean(np.abs(sbp_estimated_array - sbp_reference_array) / sbp_reference_array) * 100
        relative_error_dbp = np.mean(np.abs(dbp_estimated_array - dbp_reference_array) / dbp_reference_array) * 100

        # Mean bias (ME)
        diff_sbp = sbp_reference_array - sbp_estimated_array
        diff_dbp = dbp_reference_array - dbp_estimated_array

        me_sbp = np.mean(diff_sbp)
        me_dbp = np.mean(diff_dbp)

        # 標準偏差 (Standard deviation, SD)
        sd_sbp = np.std(sbp_estimated_array)
        sd_dbp = np.std(dbp_estimated_array)
        
        """
        # AAMI基準を満たすかどうかの判定
        determine_aami_standard(me_sbp, sd_sbp)
        determine_aami_standard(me_dbp, sd_dbp)
        
        # BHS基準の等級
        determine_bhs_grade(diff_sbp)
        determine_bhs_grade(diff_dbp)
        """
        sbp_aami = determine_aami_standard(me_sbp, sd_sbp)
        dbp_aami = determine_aami_standard(me_dbp, sd_dbp)
        
        sbp_ratios = determine_bhs_grade(diff_sbp)
        dbp_ratios = determine_bhs_grade(diff_dbp)
        
        # 相関係数 (Correlation coefficient)
        correlation_sbp = np.corrcoef(sbp_reference_array, sbp_estimated_array)[0, 1]
        correlation_dbp = np.corrcoef(dbp_reference_array, dbp_estimated_array)[0, 1]
        
        plot_filename_sbp = f"scatter_sbp_{ml_algorithm_feature.replace(' ', '_').replace('(', '').replace(')', '')}_{camera_type}.png"
        plot_filename_dbp = f"scatter_dbp_{ml_algorithm_feature.replace(' ', '_').replace('(', '').replace(')', '')}_{camera_type}.png"

        plot_scatter(sbp_reference_array, sbp_estimated_array, plot_filename_sbp)
        plot_scatter(dbp_reference_array, dbp_estimated_array, plot_filename_dbp)

        print(f"Scatter plot saved: {plot_filename_sbp}")
        print(f"Scatter plot saved: {plot_filename_dbp}")
        
        print("Correlation coefficient of SBP ({}): {:.3f}".format(ml_algorithm_feature, correlation_sbp))
        print("Correlation coefficient of DBP ({}): {:.3f}".format(ml_algorithm_feature, correlation_dbp))
                
        results.append((ml_algorithm_feature, mae_sbp, mae_dbp, relative_error_sbp, relative_error_dbp, me_sbp, me_dbp, sd_sbp, sd_dbp, sbp_aami, dbp_aami, sbp_ratios, dbp_ratios, correlation_sbp, correlation_dbp))
    
    file_exists = os.path.exists("mae_results.txt")
    with open("mae_results.txt", "a") as file:
        file.write(f"Results for {camera_type}:\n")
        for result in results:
            file.write(f"{result[0]}\n")
            file.write(f"MAE of SBP: {result[1]:.3f} [mmHg]\n")
            file.write(f"MAE of DBP: {result[2]:.3f} [mmHg]\n")
            file.write(f"Relative Error of SBP: {result[3]:.3f} [%]\n")
            file.write(f"Relative Error of DBP: {result[4]:.3f} [%]\n")
            file.write(f"ME of SBP: {result[5]:.3f} [mmHg]\n")
            file.write(f"ME of DBP: {result[6]:.3f} [mmHg]\n")
            file.write(f"SD of SBP: {result[7]:.3f} [mmHg]\n")
            file.write(f"SD of DBP: {result[8]:.3f} [mmHg]\n")
            file.write(f"AAMI of SBP: {result[9]:.3f} [mmHg]\n")
            file.write(f"AAMI of DBP: {result[10]:.3f} [mmHg]\n")
            file.write(f"SBP ratios less than 5, 10, 15: {result[11][0]:.2f}, {result[11][1]:.2f}, {result[11][2]:.2f}\n")
            file.write(f"DBP ratios less than 5, 10, 15: {result[12][0]:.2f}, {result[12][1]:.2f}, {result[12][2]:.2f}\n")
            file.write(f"BHS grade for SBP: {result[11][3]}\n")
            file.write(f"BHS grade for DBP: {result[12][3]}\n")
            file.write(f"Correlation coefficient of SBP: {result[13]:.3f}\n")
            file.write(f"Correlation coefficient of DBP: {result[14]:.3f}\n")
            file.write("\n") 
        # 選択された特徴量を追加
        if selected_features_sbp:
            file.write(f"Selected Features for SBP: {selected_features_sbp}\n")
        if selected_features_dbp:
            file.write(f"Selected Features for DBP: {selected_features_dbp}\n")
        file.write("\n")

    return mae_list


def determine_aami_standard(me_bp, sd_bp):
    # ME (mean bias) を絶対値にする
    me_bp_abs = abs(me_bp)

    print("ME and SD: {:.3f} {:.3f}".format(me_bp, sd_bp))

    # MEが5[mmHg]以下，SD (standard deviation) が8[mmHg]以下なら基準を満たす
    if me_bp_abs <= 5 and sd_bp <= 8:
        print("comply with AAMI standards")
        return True
    else:
        print("does not comply with AAMI standards")
        return False

def plot_scatter(bp_reference, bp_estimated, imgname_save):
    # 血圧の計測値と予測値についての散布図のプロット・保存
    plt.scatter(bp_reference, bp_estimated)
    plt.savefig(imgname_save)
    plt.close()


def determine_bhs_grade(diff_bp):
    # diffを絶対値にする
    diff_bp_abs = abs(diff_bp)

    # <5, <10, <15 [mmHg]の割合を求める
    ratio_less_than_5 = np.sum(diff_bp_abs < 5) / len(diff_bp_abs)
    ratio_less_than_10 = np.sum(diff_bp_abs < 10) / len(diff_bp_abs)
    ratio_less_than_15 = np.sum(diff_bp_abs < 15) / len(diff_bp_abs)

    print("ratio less than 5, 10, 15: {:.2f} {:.2f} {:.2f}".format(ratio_less_than_5, ratio_less_than_10,
                                                                   ratio_less_than_15))

    # A, B, C等級の条件分岐 (C等級に満たない場合はD等級)
    grade = "D"
    if ratio_less_than_5 >= 0.6 and ratio_less_than_10 >= 0.85 and ratio_less_than_15 >= 0.95:
        grade = "A"
        print("BHS grade: A")
    elif ratio_less_than_5 >= 0.5 and ratio_less_than_10 >= 0.75 and ratio_less_than_15 >= 0.90:
        grade = "B"
        print("BHS grade: B")
    elif ratio_less_than_5 >= 0.4 and ratio_less_than_10 >= 0.76 and ratio_less_than_15 >= 0.85:
        grade = "C"
        print("BHS grade: C")
    else:
        print("BHS grade: D")
        
    return ratio_less_than_5, ratio_less_than_10, ratio_less_than_15, grade


def main():
    dir_load_name = "E:\\2024_ex\\data_10s_LOSO\\"
    dir_load_suffix = ""  # 読み込み用ディレクトリの指定に用いる（任意，用いない場合は""で設定）
    dir_features = "rgb-nir\\features\\"
    # filepaths = natsorted(glob.glob(dir_load_name + "*" + dir_load_suffix))
    filepaths = [d for d in natsorted(glob.glob(os.path.join(dir_load_name, '*'))) if os.path.isdir(d)]
    FEATURES_NUM_CN =29
    FEATURES_NUM_DR = 22
    FEATURES_NUM_ALL = FEATURES_NUM_CN + FEATURES_NUM_DR
    num_experiments = len(filepaths)  # 被験者数
    
    filter_flag = False
    thres_age = 30
    ages = [50, 22, 33, 20, 20, 24, 24, 24, 24, 24, 24, 21, 50, 24, 22, 24, 24, 24, 22, 41, 60, 24, 22, 65, 24, 65, 57, 56, 20, 59, 20]


    # 血圧の正解値の読み込み (被験者名の順番がアルファベット順に従うようにする)
    
    y1 = np.loadtxt("bp_measured_ori\\araki.csv", delimiter=",", encoding="utf-8-sig")
    y2 = np.loadtxt("bp_measured_ori\\endou.csv", delimiter=",", encoding="utf-8-sig")
    y3 = np.loadtxt("bp_measured_ori\\fukai.csv", delimiter=",", encoding="utf-8-sig")
    y4 = np.loadtxt("bp_measured_ori\\fukuda.csv", delimiter=",", encoding="utf-8-sig")
    y5 = np.loadtxt("bp_measured_ori\\gotou.csv", delimiter=",", encoding="utf-8-sig")
    y6 = np.loadtxt("bp_measured_ori\\hayashi.csv", delimiter=",", encoding="utf-8-sig")
    y7 = np.loadtxt("bp_measured_ori\\igai.csv", delimiter=",", encoding="utf-8-sig")
    y8 = np.loadtxt("bp_measured_ori\\imai.csv", delimiter=",", encoding="utf-8-sig")
    y9 = np.loadtxt("bp_measured_ori\\imamura.csv", delimiter=",", encoding="utf-8-sig")
    y10 = np.loadtxt("bp_measured_ori\\ishida.csv", delimiter=",", encoding="utf-8-sig")
    y11 = np.loadtxt("bp_measured_ori\\ishiguro.csv", delimiter=",", encoding="utf-8-sig")
    y12 = np.loadtxt("bp_measured_ori\\itaya.csv", delimiter=",", encoding="utf-8-sig")
    y13 = np.loadtxt("bp_measured_ori\\izuhara.csv", delimiter=",", encoding="utf-8-sig")
    y14 = np.loadtxt("bp_measured_ori\\kaneko.csv", delimiter=",", encoding="utf-8-sig")
    y15 = np.loadtxt("bp_measured_ori\\kanndori.csv", delimiter=",", encoding="utf-8-sig")
    y16 = np.loadtxt("bp_measured_ori\\kawa.csv", delimiter=",", encoding="utf-8-sig")
    y17 = np.loadtxt("bp_measured_ori\\kinefuchi.csv", delimiter=",", encoding="utf-8-sig")
    y18 = np.loadtxt("bp_measured_ori\\kitabayashi.csv", delimiter=",", encoding="utf-8-sig")
    y19 = np.loadtxt("bp_measured_ori\\kiyohara.csv", delimiter=",", encoding="utf-8-sig")
    y20 = np.loadtxt("bp_measured_ori\\kobayashi.csv", delimiter=",", encoding="utf-8-sig")
    y21 = np.loadtxt("bp_measured_ori\\kumi.csv", delimiter=",", encoding="utf-8-sig")
    y22 = np.loadtxt("bp_measured_ori\\kurisaka.csv", delimiter=",", encoding="utf-8-sig")
    y23 = np.loadtxt("bp_measured_ori\\murai.csv", delimiter=",", encoding="utf-8-sig")
    y24 = np.loadtxt("bp_measured_ori\\ojima.csv", delimiter=",", encoding="utf-8-sig")
    y25 = np.loadtxt("bp_measured_ori\\satou.csv", delimiter=",", encoding="utf-8-sig")
    y26 = np.loadtxt("bp_measured_ori\\takahashi.csv", delimiter=",", encoding="utf-8-sig")
    y27 = np.loadtxt("bp_measured_ori\\tsumura.csv", delimiter=",", encoding="utf-8-sig")
    y28 = np.loadtxt("bp_measured_ori\\usami.csv", delimiter=",", encoding="utf-8-sig")
    y29 = np.loadtxt("bp_measured_ori\\usamikun.csv", delimiter=",", encoding="utf-8-sig")
    y30 = np.loadtxt("bp_measured_ori\\yamamoto.csv", delimiter=",", encoding="utf-8-sig")
    y31 = np.loadtxt("bp_measured_ori\\yata.csv", delimiter=",", encoding="utf-8-sig")
    

    # bp_array = np.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])
    bp_array = np.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31])
    # bp_array = np.concatenate([y2, y4, y5, y6, y7, y8, y9, y10, y11, y12, y14, y15, y16, y17, y18, y19, y22, y23, y25, y29, y31])
    

    # 被験者の状態と撮影回数をまとめたリスト（ファイル名の指定時に使用）
    # state_cnt_append_list = ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"]
    # num_state_count = len(state_cnt_append_list)
    
    """
    #5s  ex: 70 nb: 74
    subject_state_counts = {
        "araki": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  6 3 3
        "endou": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  12 6 6
        "fukai": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  18 9 9
        "fukuda": ["_ex_01", "_ex_02", "_ex_03", "_nb_02", "_nb_03"], # 5 3 2  23 12 11
        "gotou": ["_ex_01", "_ex_03"], # 2 2 0  25 14 11
        "hayashi": ["_ex_01", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 5 2 3  30 16 14
        "igai": ["_ex_01", "_ex_02", "_ex_03", "_nb_01"],  # 4 3 1  34 19 15
        "imai": ["_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 5 2 3  39 21 18
        "imamura": ["_ex_01", "_ex_02", "_nb_02", "_nb_03"],  # 4 2 2  43 23 20
        "ishida": ["_ex_02", "_ex_03", "_nb_02", "_nb_03"], # 4 2 2  47 25 22
        "ishiguro": ["_ex_01", "_ex_02", "_nb_01", "_nb_02", "_nb_03"], # 5 2 3  52 27 25
        "itaya": ["_ex_01", "_ex_02", "_nb_02", "_nb_03"], # 4 2 2  56 29 27
        "izuhara": ["_ex_02", "_nb_01", "_nb_02", "_nb_03"],  # 4 1 3  60 30 30
        "kaneko": ["_nb_01", "_nb_02", "_nb_03"],  # 3 0 3  63 30 33
        "kanndori": ["_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 5 2 3  68 32 36 
        "kawa": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02"], # 5 3 2  73 35 38 
        "kinefuchi": ["_ex_02", "_nb_01", "_nb_02", "_nb_03"], # 4 1 3  77 36 41
        "kitabayashi": ["_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 4 1 3  81 37 44
        "kiyohara": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  87 40 47
        "kobayashi": ["_ex_01", "_ex_03", "_nb_02"], # 3 2 1 90 42 48
        "kumi": ["_ex_01", "_ex_02", "_ex_03", "_nb_02", "_nb_03"],# 5 3 2  95 45 50
        "kurisaka": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  101 48 53
        "murai": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3 107 51 56
        "ojima": ["_ex_01", "_ex_03", "_nb_01", "_nb_03"],  # 4 2 2  111 53 58
        "satou": ["_ex_01", "_ex_02", "_nb_02", "_nb_03"],# 4 2 2  115 55 60
        "takahashi": ["_ex_03", "_nb_03"],# 2 1 1  117 56 61
        "tsumura": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  123 59 64
        "usami": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  129 62 67
        "usamikun": ["_ex_01", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 5 2 3  134 64 70
        "yamamoto": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  140 67 73
        "yata": ["_ex_01", "_ex_02", "_ex_03", "_nb_03"]# 4 3 1  144 70 74
    }
    
    num_state_count = [6, 6, 6, 5, 2, 5, 4, 5, 4, 4, 5, 4, 4, 3, 5, 5, 4, 4, 6, 3, 5, 6, 6, 4, 4, 2, 6, 6, 5, 6, 4]
    """
    
    #10s  ex:  nb: 
    subject_state_counts = {
        "araki": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  6 3 3
        "endou": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  12 6 6
        "fukai": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  18 9 9
        "fukuda": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  24 12 12
        "gotou": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_03"], # 5 3 2  29 15 14
        "hayashi": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  35 18 17
        "igai": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  41 21 20
        "imai":  ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  47 24 23
        "imamura": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  53 27 26
        "ishida": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  59 30 29
        "ishiguro": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  65 33 32
        "itaya": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  71 36 35
        "izuhara": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  77 39 38
        "kaneko": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  83 42 41
        "kanndori": ["_ex_01", "_ex_02", "_nb_01", "_nb_02", "_nb_03"],# 5 2 3  88 44 44 
        "kawa": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02"], # 5 3 2  93 47 46 
        "kinefuchi": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  99 50 49
        "kitabayashi":["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  105 53 52
        "kiyohara": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02"],# 5 3 2  110 56 54
        "kobayashi": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3 116 59 57
        "kumi": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  122 62 60
        "kurisaka": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  128 65 63
        "murai": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3 134 68 66
        "ojima": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  140 71 69
        "satou": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  146 74 72
        "takahashi": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  152 77 75
        "tsumura": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],# 6 3 3  158 80 78
        "usami": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  164 83 81
        "usamikun":["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"],  # 6 3 3  170 86 84
        "yamamoto": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"], # 6 3 3  176 89 87
        "yata": ["_ex_01", "_ex_02", "_ex_03", "_nb_01", "_nb_02", "_nb_03"]# 6 3 3  182 92 90
    }
    
    num_state_count = [6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    

    # 特徴量を格納するNumPy配列の用意
    # 被験者数×状態・撮影回数×特徴量の行数
    # feature_array_industrial_cam = np.empty((num_experiments, num_state_count, FEATURES_NUM_ALL + 1))
    # feature_array_rgb_nir_g = feature_array_industrial_cam.copy()
    # feature_array_rgb_nir_nir = feature_array_industrial_cam.copy()
    
    
    # RGBとNIRをまとめる場合，特徴量の数が2倍（GとNIRそれぞれの分）になるため，FEATURES_NUM_ALL*2にする
    # feature_array_rgb_nir = np.empty((num_experiments, num_state_count, FEATURES_NUM_ALL * 2 + 1))
    
    feature_array_rgb_nir_g = [np.empty((count, FEATURES_NUM_ALL + 1)) for count in num_state_count]
    feature_array_rgb_nir_nir = [np.empty((count, FEATURES_NUM_ALL + 1)) for count in num_state_count]
    feature_array_rgb_nir = [np.empty((count, FEATURES_NUM_ALL * 2 + 1)) for count in num_state_count]

    # 平均絶対誤差（mean absolute error , MAE）の結果をまとめて表示するためのNumPy配列
    #mae_array = np.empty((4, 8))
    #mae_array = np.empty((3, 20))
    mae_array = np.empty((3, 12))

    # 特徴量を読み込み，各被験者ごとに格納
    for subject_idx, filepath in enumerate(filepaths):
        # 特徴量のディレクトリ名の指定
        #subject_path = filepath.split("\\")[-1]

        # subject_pathが "subject_name + dir_load_suffix" の形式になるため，
        # dir_load_suffixを空の文字列に置換してsubject_nameを取得する
        #subject_name = subject_path.replace(dir_load_suffix, "")
        
        # ディレクトリパスから被験者名を取得
        subject_name = os.path.basename(filepath)
        state_cnt_list = subject_state_counts.get(subject_name, []) 

        for state_cnt_idx, state_cnt_append in enumerate(state_cnt_list):
            #dirname_base = filepath + "\\" + subject_name + state_cnt_append
            dirname_base = os.path.join(filepath, subject_name + state_cnt_append)
            print(dirname_base)
            #dirname_load_features_industrial_cam = dirname_base + "\\industrial_cam\\" + dir_features
            #dirname_load_features_rgb_nir = dirname_base + "\\rgb-nir\\" + dir_features
            dirname_load_features_rgb_nir = os.path.join(dirname_base, "rgb-nir", dir_features)
            print(dirname_load_features_rgb_nir)

            # 特徴量のファイル名の指定
            #filename_features_industrial_cam_cn = dirname_load_features_industrial_cam + "features_cn_g.csv"
            #filename_features_industrial_cam_dr = dirname_load_features_industrial_cam + "features_dr_g.csv"
            filename_features_rgb_nir_cn_g = dirname_load_features_rgb_nir + "short_features_cn_g.csv"
            filename_features_rgb_nir_cn_nir = dirname_load_features_rgb_nir + "short_features_cn_nir.csv"
            filename_features_rgb_nir_dr_g = dirname_load_features_rgb_nir + "short_features_dr_g.csv"
            filename_features_rgb_nir_dr_nir = dirname_load_features_rgb_nir + "short_features_dr_nir.csv"
   
            
            try:
                # 特徴量の読み込み
                features_rgb_nir_cn_g = np.loadtxt(filename_features_rgb_nir_cn_g, delimiter=",")
                features_rgb_nir_cn_nir = np.loadtxt(filename_features_rgb_nir_cn_nir, delimiter=",")
                features_rgb_nir_dr_g = np.loadtxt(filename_features_rgb_nir_dr_g, delimiter=",")
                features_rgb_nir_dr_nir = np.loadtxt(filename_features_rgb_nir_dr_nir, delimiter=",")
            except FileNotFoundError:
                print(f"File not found for {subject_name} {state_cnt_append}. Skipping.")
                continue

            # 特徴量の結合（産業用カメラ，RGB-NIRごと）
            #features_industrial_cam = np.concatenate([features_industrial_cam_cn, features_industrial_cam_dr])
            features_rgb_nir_g = np.concatenate([features_rgb_nir_cn_g, features_rgb_nir_dr_g])
            features_rgb_nir_nir = np.concatenate([features_rgb_nir_cn_nir, features_rgb_nir_dr_nir])
            features_rgb_nir = np.concatenate([features_rgb_nir_g, features_rgb_nir_nir])
            
            #年齢を特徴量に追加
            age_feature = ages[subject_idx]   # 6行の列ベクトルを作成
            features_rgb_nir_g = np.hstack([features_rgb_nir_g, age_feature])
            features_rgb_nir_nir = np.hstack([features_rgb_nir_nir, age_feature])
            features_rgb_nir = np.hstack([features_rgb_nir, age_feature])

            # 各行ごとに標準化（平均0，分散1にする）
            #features_industrial_cam = zscore(features_industrial_cam)
            features_rgb_nir_g = zscore(features_rgb_nir_g)
            features_rgb_nir_nir = zscore(features_rgb_nir_nir)
            features_rgb_nir = zscore(features_rgb_nir)

            # 特徴量を格納
            #feature_array_industrial_cam[subject_idx, state_cnt_idx] = features_industrial_cam
            # feature_array_rgb_nir_g[subject_idx, state_cnt_idx] = features_rgb_nir_g
            # feature_array_rgb_nir_nir[subject_idx, state_cnt_idx] = features_rgb_nir_nir
            # feature_array_rgb_nir[subject_idx, state_cnt_idx] = features_rgb_nir
            
            feature_array_rgb_nir_g[subject_idx][state_cnt_idx] = features_rgb_nir_g
            feature_array_rgb_nir_nir[subject_idx][state_cnt_idx] = features_rgb_nir_nir
            feature_array_rgb_nir[subject_idx][state_cnt_idx] = features_rgb_nir

            
    #     print(filepath + " is loaded!")
    # print()

    # 血圧推定結果の表示
    #print("Industrial camera (only G)")
    #mae_array[0] = estimate_bp(feature_array_industrial_cam, bp_array, num_experiments)
    #print()
    
    filtered_features_rgb_nir_g = []
    filtered_features_rgb_nir_nir = []
    filtered_features_rgb_nir = []
    filtered_bp_array = []
    
    filtered_num_experiments = 0
    
    if filter_flag:      
        for subject_idx in range(num_experiments):
            # 年齢列を確認して範囲内のデータのみ保持
            age = ages[subject_idx]  # 0番目の状態から年齢を取得
            if thres_age <= age :
                print("add subject")
                print(subject_idx)
                filtered_num_experiments += 1
                filtered_features_rgb_nir_g.append(feature_array_rgb_nir_g[subject_idx])
                filtered_features_rgb_nir_nir.append(feature_array_rgb_nir_nir[subject_idx])
                filtered_features_rgb_nir.append(feature_array_rgb_nir[subject_idx])
                filtered_bp_array.append(bp_array[6 * subject_idx:6 * subject_idx + 6])  # 正解値もフィルタリング
        
        filtered_features_rgb_nir_g = np.array(filtered_features_rgb_nir_g)
        filtered_features_rgb_nir_nir = np.array(filtered_features_rgb_nir_nir)
        filtered_features_rgb_nir = np.array(filtered_features_rgb_nir)
        filtered_bp_array = np.concatenate(filtered_bp_array, axis=0)
    

        print("RGB-NIR camera (only G)")
        #mae_array[0] = estimate_bp(feature_array_rgb_nir_g, bp_array, num_experiments)
        #mae_array[0], sbp_predicted_g, dbp_predicted_g = estimate_bp(filtered_features_rgb_nir_g, filtered_bp_array, filtered_num_experiments, "RGB-NIR G")
        print()
    
        print("RGB-NIR camera (only NIR)")
        #mae_array[1] = estimate_bp(feature_array_rgb_nir_nir, bp_array, num_experiments)
        #mae_array[1], sbp_predicted_nir, dbp_predicted_nir = estimate_bp(filtered_features_rgb_nir_nir, filtered_bp_array, filtered_num_experiments, "RGB-NIR NIR")
        print()
        """
        sbp_predicted_g_reshaped = sbp_predicted_g.reshape(filtered_features_rgb_nir.shape[0], filtered_features_rgb_nir.shape[1], 1)
        dbp_predicted_g_reshaped = dbp_predicted_g.reshape(filtered_features_rgb_nir.shape[0], filtered_features_rgb_nir.shape[1], 1)
        sbp_predicted_nir_reshaped = sbp_predicted_nir.reshape(filtered_features_rgb_nir.shape[0], filtered_features_rgb_nir.shape[1], 1)
        dbp_predicted_nir_reshaped = dbp_predicted_nir.reshape(filtered_features_rgb_nir.shape[0], filtered_features_rgb_nir.shape[1], 1)
        
        predicted_features = np.concatenate(
            [sbp_predicted_g_reshaped, dbp_predicted_g_reshaped, sbp_predicted_nir_reshaped, dbp_predicted_nir_reshaped],
            axis=2
        )
        
        predicted_features = zscore(predicted_features, axis=None)

        
        # 元の特徴量と新しい予測値を結合
        filtered_features_rgb_nir = np.concatenate([filtered_features_rgb_nir, predicted_features], axis=2)
        """
        
        print("RGB-NIR camera (G & NIR)")
        mae_array[2], sbp_predicted_gnir, dbp_predicted_gnir = estimate_bp(
            filtered_features_rgb_nir, filtered_bp_array, filtered_num_experiments, "RGB-NIR G&NIR"
        )
    
        print("RGB-NIR camera (G & NIR)")
        #mae_array[2] = estimate_bp(feature_array_rgb_nir, bp_array, num_experiments)
        mae_array[2], sbp_predicted, dbp_predicted = estimate_bp(filtered_features_rgb_nir, filtered_bp_array, filtered_num_experiments, "RGB-NIR G&NIR", num_state_count)
        print()
    else:
        print("RGB-NIR camera (only G)")
        #mae_array[0] = estimate_bp(feature_array_rgb_nir_g, bp_array, num_experiments)
        #mae_array[0], sbp_predicted_g, dbp_predicted_g = estimate_bp(feature_array_rgb_nir_g, bp_array, num_experiments, "RGB-NIR G", num_state_count)
        print()
    
        print("RGB-NIR camera (only NIR)")
        #mae_array[1] = estimate_bp(feature_array_rgb_nir_nir, bp_array, num_experiments)
        #mae_array[1], sbp_predicted_nir, dbp_predicted_nir = estimate_bp(feature_array_rgb_nir_nir, bp_array, num_experiments, "RGB-NIR NIR", num_state_count)
        print()
        
        """
        sbp_predicted_g_reshaped = sbp_predicted_g.reshape(feature_array_rgb_nir.shape[0], feature_array_rgb_nir.shape[1], 1)
        dbp_predicted_g_reshaped = dbp_predicted_g.reshape(feature_array_rgb_nir.shape[0], feature_array_rgb_nir.shape[1], 1)
        sbp_predicted_nir_reshaped = sbp_predicted_nir.reshape(feature_array_rgb_nir.shape[0], feature_array_rgb_nir.shape[1], 1)
        dbp_predicted_nir_reshaped = dbp_predicted_nir.reshape(feature_array_rgb_nir.shape[0], feature_array_rgb_nir.shape[1], 1)
        
        predicted_features = np.concatenate(
            [sbp_predicted_g_reshaped, dbp_predicted_g_reshaped, sbp_predicted_nir_reshaped, dbp_predicted_nir_reshaped],
            axis=2
        )
        
        predicted_features = zscore(predicted_features, axis=None)
        
        # 元の特徴量と新しい予測値を結合
        feature_array_rgb_nir = np.concatenate([feature_array_rgb_nir, predicted_features], axis=2)
        """
    
        print("RGB-NIR camera (G & NIR)")
        #mae_array[2] = estimate_bp(feature_array_rgb_nir, bp_array, num_experiments)
        mae_array[2], sbp_predicted_gnir, dbp_predicted_gnir = estimate_bp(feature_array_rgb_nir, bp_array, num_experiments, "RGB-NIR G&NIR", num_state_count)
        print()
    
    # 推定値の保存用辞書を準備
    all_predictions = {
        "sbp_predicted_g": sbp_predicted_g.tolist(),
        "dbp_predicted_g": dbp_predicted_g.tolist(),
        "sbp_predicted_nir": sbp_predicted_nir.tolist(),
        "dbp_predicted_nir": dbp_predicted_nir.tolist(),
        "sbp_predicted_gnir": sbp_predicted_gnir.tolist(),
        "dbp_predicted_gnir": dbp_predicted_gnir.tolist()
    }
    
    # 推定値をファイルに保存
    save_predictions_to_txt(all_predictions)
    
    

    # 列および行のラベルを設定
    #columns = ["G (industrial)", "G (RGB-NIR)", "NIR (RGB-NIR)", "G&NIR"]  # G(産業用カメラ)，G(RGB-NIRカメラ，以下同様)
    #index = ["RF  (all features) SBP", "RF  (all features) DBP", "RF  (  selected  ) SBP", "RF  (  selected  ) DBP",
    #        "SVR (all features) SBP", "SVR (all features) DBP", "SVR (  selected  ) SBP", "SVR (  selected  ) DBP"]
    columns = ["G (RGB-NIR)", "NIR (RGB-NIR)", "G&NIR"]  # G(産業用カメラ)，G(RGB-NIRカメラ，以下同様)

    index = ["RF  (all features) SBP", "RF  (all features) DBP", "RF  (selected features) SBP", "RF  (selected features) DBP",
             "SVR (all features) SBP", "SVR (all features) DBP", "SVR (selected features) SBP", "SVR (selected features) DBP",
             "LightGBM (all features) SBP", "LightGBM (all features) DBP", "LightGBM (selected features) SBP", "LightGBM (selected features) DBP"]

    # NumPy配列をpd.DataFrameに変換し、列と行にラベルを設定
    mae_array = mae_array.T
    mae_df = pd.DataFrame(mae_array, columns=columns, index=index)
    
    print(mae_df)


if __name__ == "__main__":
    main()
