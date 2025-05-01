import numpy as np
import optuna
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from .model_factory import get_model
from .optuna_objectives import objective_rf, objective_svr, objective_lgbm
from ..evaluation.metrics import calc_metrics

from ..save_utils.save_utils import save_model, save_selected_features, save_best_params
import yaml

def load_model_params(path="c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\blood_pressure_estimation_project\\config\\model_params.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_cv_splits(method, num_subjects, n_splits=5):
    
    indices = np.arange(num_subjects)

    if method == "LOSO":
        return [(np.delete(indices, i), [i]) for i in indices]

    elif method == "KFold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(kf.split(indices))

    else:
        raise ValueError(f"Unsupported method: {method}")

def perform_machine_learning(X_train, X_test, y_train_sbp, y_train_dbp,model_list,use_optuna=True, n_features_to_select=25, rand=0):
    n_features_to_select = 25
    studies = {}
    best_params = {}

    # 外部からパラメータ読込（YAML）
    params = load_model_params()
    default_params = {
        "rf_sbp": params.get("rf", {}).get("sbp", {}),
        "rf_dbp": params.get("rf", {}).get("dbp", {}),
        "svr_sbp": params.get("svr", {}).get("sbp", {}),
        "svr_dbp": params.get("svr", {}).get("dbp", {}),
        "lgbm_sbp": params.get("lgbm", {}).get("sbp", {}),
        "lgbm_dbp": params.get("lgbm", {}).get("dbp", {}),
    }

    # Optuna最適化 or デフォルト使用
    if use_optuna:
        for target, y_train in zip(["sbp", "dbp"], [y_train_sbp, y_train_dbp]):
            for model_name in model_list:
                obj_func_map = {"rf": objective_rf, "svr": objective_svr, "lgbm": objective_lgbm}
                objective_func = obj_func_map[model_name]

                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=100)
                optuna_best_params = study.best_params

                model_optuna = get_model(model_name, optuna_best_params, rand)
                model_default = get_model(model_name, default_params[f"{model_name}_{target}"], rand)

                model_optuna.fit(X_train, y_train)
                model_default.fit(X_train, y_train)

                score_optuna = -cross_val_score(model_optuna, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
                score_default = -cross_val_score(model_default, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()

                best_params[f"{model_name}_{target}"] = optuna_best_params if score_optuna < score_default else default_params[f"{model_name}_{target}"]
    else:
        best_params = default_params

    save_best_params(best_params,output_file="best_params.txt")

    results = {}
    selected_features = {}
    trained_models = {}  
    
    for model_name in model_list:
        for target, y_train, y_test, model_key in [("sbp", y_train_sbp, y_train_sbp, f"{model_name}_sbp"),
                                                   ("dbp", y_train_dbp, y_train_dbp, f"{model_name}_dbp")]:
            model = get_model(model_name, best_params[model_key], rand)
            model.fit(X_train, y_train)
            pred_all = model.predict(X_test)
            save_model(model, model_name, target, "all")

            results[f"{target}_{model_name}_all"] = pred_all
            trained_models[f"{model_name}_{target}_all"] = model  # ✅ モデル登録
            
            # RFEも適用
            rfe_model = get_model(model_name, best_params[model_key], rand)
            rfe = RFE(rfe_model, n_features_to_select=n_features_to_select)
            X_train_rfe = rfe.fit_transform(X_train, y_train)
            X_test_rfe = rfe.transform(X_test)
            rfe_model.fit(X_train_rfe, y_train)
            pred_rfe = rfe_model.predict(X_test_rfe)

            selected_indices = [i for i, b in enumerate(rfe.support_) if b]
            save_model(rfe_model, model_name, target, "rfe")
            save_selected_features( model_name, target.upper(), selected_indices)

            results[f"{target}_{model_name}_rfe"] = pred_rfe
            trained_models[f"{model_name}_{target}_rfe"] = rfe_model  # ✅ モデル登録
            selected_features[f"{target}_{model_name}"] = selected_indices

    return results, selected_features, trained_models

def estimate_bp(feature_array, bp_array, num_subjects, num_experiments,
                model_list, use_optuna=True, n_features_to_select=25,cv_method="LOSO",n_splits=5):
    
    sbp_ref, dbp_ref = [], []

    # 予測結果格納用の辞書
    results_all = {f"{bp}_{model}_{sel}": []
                   for bp in ["sbp", "dbp"]
                   for model in model_list
                   for sel in ["all", "rfe"]}

    # 特徴量選択インデックス
    selected_features_sbp_count = [0] * feature_array.shape[2]
    selected_features_dbp_count = [0] * feature_array.shape[2]
    
    cv_splits = get_cv_splits(method="LOSO", num_subjects=num_subjects, n_splits=n_splits)
    
        # 予測結果格納用の辞書
    results_all = {f"{bp}_{model}_{sel}": []
                   for bp in ["sbp", "dbp"]
                   for model in model_list
                   for sel in ["all", "rfe"]}

    # 特徴量選択インデックスの出現頻度をカウント
    selected_features_sbp_count = [0] * feature_array.shape[2]
    selected_features_dbp_count = [0] * feature_array.shape[2]

    # 分割方法に応じたインデックスの取得
    cv_splits = get_cv_splits(method=cv_method, num_subjects=num_subjects, n_splits=n_splits)

    for train_idx, test_idx in cv_splits:
        # 特徴量の抽出
        X_train = np.concatenate([feature_array[i] for i in train_idx])
        X_test  = np.concatenate([feature_array[i] for i in test_idx])

        # 血圧データの抽出
        y_train = np.concatenate([
            bp_array[num_experiments * i: num_experiments * i + num_experiments]
            for i in train_idx
        ])
        y_test = np.concatenate([
            bp_array[num_experiments * i: num_experiments * i + num_experiments]
            for i in test_idx
        ])

        y_train_sbp, y_train_dbp = y_train[:, 0], y_train[:, 1]
        y_test_sbp, y_test_dbp = y_test[:, 0], y_test[:, 1]

        # NaN補完（各特徴量の平均で）
        X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
        X_test  = np.nan_to_num(X_test, nan=np.nanmean(X_test))

        # モデル学習・推定（外部関数に委譲）
        preds, selected, trained_models = perform_machine_learning(
            X_train, X_test, y_train_sbp, y_train_dbp,
            model_list=model_list,
            use_optuna=use_optuna,
            n_features_to_select=n_features_to_select
        )

        # 正解値を記録
        sbp_ref.extend(y_test_sbp)
        dbp_ref.extend(y_test_dbp)

        # 予測値を記録
        for key, val in preds.items():
            results_all[key].extend(val)

        # 特徴量選択の集計
        for model_name in model_list:
            sbp_key = f"sbp_{model_name}"
            dbp_key = f"dbp_{model_name}"

            for i in selected.get(sbp_key, []):
                selected_features_sbp_count[i] += 1

            for i in selected.get(dbp_key, []):
                selected_features_dbp_count[i] += 1

    # 評価用配列
    estimated_arrays = []
    ml_algorithm_feature_list = []
    for model_name in model_list:
        for sel in ["all", "rfe"]:
            estimated_arrays.append(np.array(results_all[f"sbp_{model_name}_{sel}"]))
            estimated_arrays.append(np.array(results_all[f"dbp_{model_name}_{sel}"]))
            ml_algorithm_feature_list.append(f"{model_name}_{sel}")
    
    selected_renamed = {}
    for old_key, indices in selected.items():
        target, model = old_key.split("_")  
        new_key = f"{model}_{target}_rfe"   
        selected_renamed[new_key] = indices
        
    mae_list = calc_metrics(
        np.array(sbp_ref), np.array(dbp_ref), estimated_arrays,
        selected_features_sbp_count, selected_features_dbp_count,ml_algorithm_feature_list,selected_renamed
    )
    
    predictions_dict = {}
    for model_name in model_list:
        for sel in ["all", "rfe"]:
            key_sbp = f"sbp_{model_name}_{sel}"
            key_dbp = f"dbp_{model_name}_{sel}"
            predictions_dict[key_sbp] = np.array(results_all.get(key_sbp, []))
            predictions_dict[key_dbp] = np.array(results_all.get(key_dbp, []))
            


    return mae_list,predictions_dict,selected_features_sbp_count,selected_features_dbp_count,trained_models,selected_renamed

