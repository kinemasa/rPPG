import numpy as np
from data_loader.load_features import load_features_with_age,load_all_data,df_to_np_arrays
from ml_models.trainer import estimate_bp
from save_utils.save_utils import save_predictions_to_txt,save_model,save_metrics_to_txt,save_all_models
from evaluation.metrics import calc_metrics
import yaml
import os

def main():
    # -----------------------------
    # 設定ファイルの読み込み
    # -----------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))  # main.pyのあるディレクトリ
    config_path = os.path.join(base_dir, "config", "config.yaml")
    config_age_path =os.path.join(base_dir, "config", "config_ages.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    with open(config_age_path,"r",encoding="utf-8") as f:
        config_age = yaml.safe_load(f)
        
        


    feature_root = config["feature_root"]
    bp_root = config["bp_folder_root"]
    feature_subdir = config["feature_subdir"]
    feature_cn_num = config["feature_cn_num"]
    feature_dr_num = config["feature_dr_num"]
    model_list = config["models_to_use"]
    use_optuna = config["use_optuna"]
    n_features_to_select = config["n_features_to_select"]
    state_list = config["state_list"]
    age_dict = config_age.get("subject_ages", {})  


    # -----------------------------
    # 特徴量・正解値の読み込み
    # -----------------------------
    features_g,subject_names = load_features_with_age(
        base_dir=feature_root,
        feature_dir=feature_subdir,
        state_list=state_list,
        feature_num_cn=feature_cn_num,
        feature_num_dr=feature_dr_num,
        age_dict=age_dict
    )
    df = load_all_data(feature_root,feature_subdir,bp_root, config_age_path, feature_cn_num, feature_dr_num)
    num_subjects = len(subject_names)
    
    # print(df)
    feature_g, bp_array =df_to_np_arrays(df, feature_prefix="feature_", num_trials_per_subject=3)
    # print(feature_g)
    # -----------------------------
    # 推定・評価（G, NIR, G+NIR それぞれ）
    # -----------------------------
    
    # print("[INFO] Estimating with RGB-NIR G only")
    mae_list,predictions_dict,selected_features_sbp_count,selected_features_dbp_count,trained_model= estimate_bp(features_g, bp_array, num_subjects,3,
                                        model_list, use_optuna, n_features_to_select)
    
    print(mae_list)
    
    # # -----------------------------
    # # 推定値保存（例：SVRの結果のみ）
    # # -----------------------------
    save_predictions_to_txt({
    key: val.tolist() for key, val in predictions_dict.items()})
    
    # # -----------------------------
    # # 評価指標の計算・保存
    # # -----------------------------
    save_metrics_to_txt(mae_list)
    # # -----------------------------
    # # モデルの保存
    # # -----------------------------
    save_all_models(trained_model,"saved_models")
    print("[DONE] All processing completed.")


if __name__ == "__main__":
    main()
