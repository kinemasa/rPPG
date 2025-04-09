import os
import joblib
import pandas as pd
import pickle

def save_best_params( best_params, output_file="best_params.txt"):
    """
    モデルごとの最適パラメータをテキストファイルに保存する

    Parameters
    ----------
    best_params : dict
        モデル名とパラメータの辞書（例: {"rf_sbp": {...}, "svr_dbp": {...}}）
    output_file : str
        保存先ファイル名
    """
    dir_name = os.path.dirname(output_file)
    if dir_name:  # 空でない場合のみディレクトリ作成
        os.makedirs(dir_name, exist_ok=True)
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a") as f:
        for model_key, params in best_params.items():
            f.write(f"{model_key}:\n")
            for param_name, value in params.items():
                f.write(f"  {param_name}: {value}\n")
            f.write("\n")


def save_model(model, model_name, target, select,save_dir="saved_models"):
    """
    学習済みモデルをファイルに保存（joblib）

    Parameters
    ----------
    model : object
        学習済みモデル
    model_name : str
        モデルの名前（例: "rf", "svr"）
    target : str
        "sbp" or "dbp"
    select : str
        "all" or "rfe"（特徴量選択の有無）
    save_dir : str
        モデル保存先ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name}_{target}_{select}.pkl"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    
def save_all_models(models_dict, save_dir="saved_models_all"):
    """
    モデル辞書を一括でPickle形式で保存する。

    Parameters
    ----------
    models_dict : dict
        学習済みモデルの辞書。キーは "rf_sbp_all" のような形式。
    save_dir : str
        モデル保存のベースディレクトリ。
    camera_type : str
        サブディレクトリ名として使用（例：'g' など）。
    """
    # 保存ディレクトリの作成
    full_save_path = save_dir
    os.makedirs(full_save_path, exist_ok=True)

    for key, model in models_dict.items():
        file_path = os.path.join(full_save_path, f"{key}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model: {file_path}")


def save_selected_features(model_name, target, selected_features, output_file="selected_features.txt"):
    """
    選択された特徴量のインデックスを保存

    Parameters
    ----------
    model_name : str
        モデル名（例: "RandomForest"）
    target : str
        予測対象（"SBP" or "DBP"）
    selected_features : list
        選択された特徴量のインデックス
    output_file : str
        保存先ファイル名
    """
    dir_name = os.path.dirname(output_file)
    if dir_name:  # 空でなければ作成
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file, "a") as f:
        f.write(f"Model: {model_name}, Target: {target}\n")
        f.write(f"Selected Features: {selected_features}\n\n")


def save_predictions_to_txt(predictions, output_file="predictions.txt"):
    """
    推定値の保存（テキスト形式）

    Parameters
    ----------
    predictions : dict
        予測値（{"sbp_predicted": [...], "dbp_predicted": [...]}）
    output_file : str
        保存ファイル名
    """
    dir_name = os.path.dirname(output_file)
    if dir_name:  # 空でなければ作成
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file, "a") as f:
        for key, values in predictions.items():
            f.write(f"{key}:\n")
            f.write(", ".join(map(str, values)) + "\n\n")
            
def save_metrics_to_txt(mae_dict, output_file="metrics.txt"):
    """
    MAEなどの評価指標をテキスト形式で保存する。

    Parameters
    ----------
    mae_dict : dict
        モデル・特徴量ごとのMAE辞書（例: {'sbp_rf_all': 5.12, ...}）
    output_file : str
        保存ファイルパス
    """
    
    dir_name = os.path.dirname(output_file)
    if dir_name:  # 空でなければ作成
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for key, value in mae_dict.items():
            f.write(f"{key}: {value:.3f}\n")
