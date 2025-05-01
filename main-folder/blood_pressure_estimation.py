import os
import numpy as np
from pathlib import Path
from scipy import signal
import sys
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
## 脈波取得用関数
from blood_pressure_estimation_project.signal_processing.signal import extract_green_signal,bandpass_filter_pulse,detrend_signal
from blood_pressure_estimation_project.signal_processing.peak_detection import detect_pulse_peak
from blood_pressure_estimation_project.signal_processing.visualize import visualize_pulse
## 脈波特徴量取得用関数
from blood_pressure_estimation_project.signal_processing.analyze import select_pulses_by_statistics,check_sdppg_waveform,analyze_pulses
from blood_pressure_estimation_project.signal_processing.t_generation import generate_t1,generate_t2
from blood_pressure_estimation_project.signal_processing.get_feature import calc_contour_features,calc_dr_features
from mycommon.read_yaml import load_config
from mycommon.select_folder import select_folder

## 機械学習用関数
from blood_pressure_estimation_project.data_loader.load_features import load_features_with_age,load_all_data,df_to_np_arrays
from blood_pressure_estimation_project.ml_models.trainer import estimate_bp
from blood_pressure_estimation_project.save_utils.save_utils import save_predictions_to_txt,save_model,save_metrics_to_txt,save_all_models
from blood_pressure_estimation_project.evaluation.metrics import calc_metrics

def load_model(model_save_dir,model_name, target, select):
    MODEL_SAVE_DIR = model_save_dir
    file_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{target}_{select}.pkl")
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        raise FileNotFoundError(f"Model file not found: {file_path}")

def get_pulse_signal(csv_path, save_dir, bandpass_range,sampling_rate, time):
    """1ファイルに対して脈波処理・保存・可視化を実行"""
    os.makedirs(save_dir, exist_ok=True)

    # G成分抽出
    pulse_raw = extract_green_signal(csv_path)

    # 傾き除去 & バンドパスフィルタ
    pulse_detrended = detrend_signal(pulse_raw)
    pulse_filtered = bandpass_filter_pulse(pulse_detrended,bandpass_range, sampling_rate)

    # ピーク検出
    peak_indexes, valley_indexes = detect_pulse_peak(pulse_filtered, sampling_rate)

    # 可視化 & 保存
    save_img_peak = Path(save_dir) / "g_peak.png"

    visualize_pulse(pulse_filtered, save_img_peak, peak_indexes, valley_indexes, sampling_rate, time)

    # フィルタ済み脈波を保存
    save_path = Path(save_dir) / "g.csv"
    np.savetxt(save_path, pulse_filtered, delimiter=",")


def get_pulse_feature(csv_file,bandpass_range,sampling_rate,resampling_rate):
    
    pulse_bandpass_filtered = np.loadtxt(csv_file, delimiter=",")
    
    # 傾き除去
    pulse_detrended = signal.detrend(pulse_bandpass_filtered)
    # バンドパスフィルタ処理
    pulse_bandpass_filtered = bandpass_filter_pulse(pulse_detrended,bandpass_range,sampling_rate)
    
    # ピーク検出 (上側，下側)
    peak_indexes, valley_indexes = detect_pulse_peak(pulse_bandpass_filtered, sampling_rate)
    
    # 各1波形の面積，持続時間，最大振幅を格納するリスト
    area_list = []
    duration_time_list = []
    amplitude_list = []
    acceptable_idx_list = []

    ## 脈波波形の微分等の解析
    area_list, duration_time_list, amplitude_list, acceptable_idx_list,pulse_waveform_num= analyze_pulses(pulse_bandpass_filtered,valley_indexes)
    acceptable_idx_list =select_pulses_by_statistics(area_list, duration_time_list, amplitude_list, pulse_waveform_num)
    
    ## t1を求める
    t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, success = generate_t1(pulse_bandpass_filtered,valley_indexes,amplitude_list,acceptable_idx_list,resampling_rate)
    
    ## t2を求める
    t2 =generate_t2(t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, upper_ratio=0.10)
    
    # t2から特徴量を求める
    features_cn_array = calc_contour_features(t2,resampling_rate)
    features_dr_array = calc_dr_features(t2,resampling_rate)
    
    return features_cn_array,features_dr_array

def bp_estimation(feature_root,feature_subdir,state_list,feature_cn_num,feature_dr_num,age_dict,bp_root,config_age_path,model_list, use_optuna, n_features_to_select,cv_method,n_splits):
    features_g,subject_names = load_features_with_age(
        base_dir=feature_root,
        feature_dir=feature_subdir,
        state_list=state_list,
        feature_num_cn=feature_cn_num,
        feature_num_dr=feature_dr_num,
        age_dict=age_dict
    )
    ex_num = len(state_list)
    df = load_all_data(feature_root,feature_subdir,bp_root, config_age_path, feature_cn_num, feature_dr_num)
    num_subjects = len(subject_names)
    
    feature_g, bp_array =df_to_np_arrays(df, feature_prefix="feature_", num_trials_per_subject=3)
    # -----------------------------
    # 推定・評価（G, NIR, G+NIR それぞれ）
    # -----------------------------
    
    # print("[INFO] Estimating with RGB-NIR G only")
    mae_list,predictions_dict,selected_features_sbp_count,selected_features_dbp_count,trained_model,selected= estimate_bp(features_g, bp_array, num_subjects,ex_num,
                                        model_list, use_optuna, n_features_to_select,cv_method,n_splits)
    
    # # -----------------------------
    # # 推定値保存
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
    save_all_models(trained_model,selected,"saved_models")
    print("[DONE] All processing completed.")

def bp_predict_with_saved_models(feature_root, feature_subdir, state_list,
                                 feature_cn_num, feature_dr_num, age_dict,
                                 bp_root, config_age_path,
                                 model_list, selection_types, model_dir):
    # 特徴量と血圧データを読み込み
    features_g, subject_names = load_features_with_age(
        base_dir=feature_root,
        feature_dir=feature_subdir,
        state_list=state_list,
        feature_num_cn=feature_cn_num,
        feature_num_dr=feature_dr_num,
        age_dict=age_dict
    )
    df = load_all_data(feature_root, feature_subdir, bp_root, config_age_path,
                       feature_cn_num, feature_dr_num)
    
    num_subjects = len(subject_names)
    ex_num = len(state_list)
    features_np, bp_array = df_to_np_arrays(df, feature_prefix="feature_", num_trials_per_subject=ex_num)
    features_np = features_np.reshape(-1, features_np.shape[-1])
    
    age_vector = []
    for subject_name in subject_names:
        age = age_dict.get(subject_name, 0)
        age_vector.extend([age] * ex_num)  # 試行数だけ繰り返す

    # numpy配列にして shape = (num_subjects * ex_num, 1)
    age_vector = np.array(age_vector).reshape(-1, 1)

    # 特徴量に年齢を追加（shape: (N, D+1)）
    features_np = np.hstack([features_np, age_vector])
    sbp_ref, dbp_ref = bp_array[:, 0], bp_array[:, 1]
    predictions_dict = {}
    estimated_arrays = []
    ml_algorithm_feature_list = []

    for model_name in model_list:
        for sel in selection_types:
            key_sbp = f"{model_name}_sbp_{sel}"
            key_dbp = f"{model_name}_dbp_{sel}"
            sbp_model_path = os.path.join(model_dir, f"{key_sbp}.pkl")
            dbp_model_path = os.path.join(model_dir, f"{key_dbp}.pkl")

            if not os.path.exists(sbp_model_path) or not os.path.exists(dbp_model_path):
                print(f"[WARNING] モデルが見つかりません: {sbp_model_path} または {dbp_model_path}")
                continue

            # モデル読み込み
            sbp_model = joblib.load(sbp_model_path)
            dbp_model = joblib.load(dbp_model_path)
            
            # 🔻 RFEモデルの場合は選択された特徴量だけ抽出
            if sel == "rfe":
                sbp_feat_path = os.path.join(model_dir, f"{model_name}_sbp_rfe_selected_features.npy")
                dbp_feat_path = os.path.join(model_dir, f"{model_name}_dbp_rfe_selected_features.npy")

                if not os.path.exists(sbp_feat_path) or not os.path.exists(dbp_feat_path):
                    print(f"[WARNING] RFE選択特徴量が見つかりません: {sbp_feat_path}")
                    continue

                sbp_selected_idx = np.load(sbp_feat_path)
                dbp_selected_idx = np.load(dbp_feat_path)

                X_sbp = features_np[:, sbp_selected_idx]
                X_dbp = features_np[:, dbp_selected_idx]
            else:
                X_sbp = features_np
                X_dbp = features_np

            # 推定
            sbp_pred = sbp_model.predict(X_sbp)
            dbp_pred = dbp_model.predict(X_dbp)

            predictions_dict[key_sbp] = sbp_pred
            predictions_dict[key_dbp] = dbp_pred

            estimated_arrays.append(sbp_pred)
            estimated_arrays.append(dbp_pred)
            ml_algorithm_feature_list.append(f"{model_name}_{sel}")

    # 評価指標計算
    mae_list = calc_metrics(
    sbp_reference_array=sbp_ref,
    dbp_reference_array=dbp_ref,
    estimated_arrays=estimated_arrays,
    selected_features_sbp=[],  # 空でOK
    selected_features_dbp=[],  # 空でOK
    ml_algorithm_feature_list=ml_algorithm_feature_list)

    # 結果保存
    save_predictions_to_txt({k: v.tolist() for k, v in predictions_dict.items()})
    save_metrics_to_txt(mae_list)
    print("[DONE] 予測と評価が完了しました。")

    return mae_list, predictions_dict


def main():
    
    #==========config=============================
    current_path = Path(__file__)
    parent_path = current_path.parent     
    config = load_config(str(parent_path)+"\\blood_pressure_config.yaml")
    config_age_path = str(parent_path)+"\\blood_pressure_age.yaml"
    config_age =load_config(config_age_path)

    DEFAULT_BANDPASS_RANGE_HZ = config["preprocessing"]["bandpass_range_hz"]
    DEFAULT_SAMPLING_RATE     = config["preprocessing"]["sampling_rate"]
    DEFAULT_CAPTURE_TIME      = config["preprocessing"]["capture_time"]
    DEFAULT_OUTPUT_FOLDER_NAME = config["preprocessing"]["output_folder_name"]
    DEFAULT_RESAMPLING_RATE  =config["feature_extraction"]["resampling_rate"]

    feature_root = config["feature_root"]
    bp_root = config["bp_folder_root"]
    feature_subdir = config["feature_subdir"]
    feature_cn_num = config["feature_cn_num"]
    feature_dr_num = config["feature_dr_num"]
    model_list = config["models_to_use"]
    use_optuna = config["use_optuna"]
    n_features_to_select = config["n_features_to_select"]
    state_list = config["state_list"]
    ex_num  = len(state_list)
    age_dict = config_age.get("subject_ages", {})  
    cv_method = config["cv_method"]
    n_splits = config["n_splits"]
    model_dir = config["model_dir"]
    selection_types = config["selection_types"]
    #=======================================
    
    
    ##画像から取得した.csvファイルが入った親フォルダを選択する
    # INPUT_PULSE_DATA = select_folder("脈波データを取得します")
    # INPUT_BP_DATA = select_folder("血圧データを取得します")
    
    INPUT_PULSE_DATA =feature_root
    ## ===========脈波取得処理 ==================================
    input_path = Path(INPUT_PULSE_DATA)
    csv_files = []
    for subject_folder in input_path.iterdir():
        if subject_folder.is_dir():
            # 各サブフォルダの中のcsvファイルだけ探す（再帰しない）
            csv_in_subfolder = list(subject_folder.glob("*.csv"))
            csv_files.extend(csv_in_subfolder)
    
    for csv_path in csv_files:
        csv_stem = csv_path.stem
        save_dir = Path(csv_path).parent / csv_stem / "processed"
        get_pulse_signal(csv_path, save_dir,DEFAULT_BANDPASS_RANGE_HZ, DEFAULT_SAMPLING_RATE,DEFAULT_CAPTURE_TIME)
        print(f"{csv_path} is Done !!")
        
        
    ## ===========脈波特徴量取得処理 ==================================
    processed_csv_files = []

    for subject_folder in input_path.iterdir():
        if subject_folder.is_dir():
            for ex_folder in subject_folder.iterdir():
                if ex_folder.is_dir():    
                # 各被験者フォルダ
                    processed_folder = ex_folder / "processed"
                    if processed_folder.exists():
                        csvs = list(processed_folder.glob("*.csv"))
                        processed_csv_files.extend(csvs)

    for csv_path in processed_csv_files:
        features_cn_array,features_dr_array =get_pulse_feature(csv_path,DEFAULT_BANDPASS_RANGE_HZ,DEFAULT_SAMPLING_RATE,DEFAULT_RESAMPLING_RATE)
        csv_stem = csv_path.stem  # ファイル名（拡張子なし）
        dir_path = Path(csv_path).parents[1]
        print(dir_path)
        # 特徴量保存用のディレクトリ名の指定
        dir_name_save = str(dir_path)  + "\\"+ "features\\"
        os.makedirs(dir_name_save, exist_ok=True)

        # 特徴量保存用のファイル名の指定
        filename_save_features_cn = dir_name_save + "features_cn" + ".csv"
        filename_save_features_dr = dir_name_save + "features_dr" + ".csv"

        # 特徴量の保存
        np.savetxt(filename_save_features_cn, features_cn_array, delimiter=",")
        np.savetxt(filename_save_features_dr, features_dr_array, delimiter=",")
        print(csv_path, "is Done!")
        
    ## ===========機械学習 ==================================
    bp_estimation(feature_root,feature_subdir,state_list,feature_cn_num,feature_dr_num,age_dict,bp_root,config_age_path,model_list, use_optuna, n_features_to_select,cv_method,n_splits)
    # bp_predict_with_saved_models(feature_root, feature_subdir, state_list,feature_cn_num, feature_dr_num, age_dict,bp_root, config_age_path,model_list, selection_types, model_dir)


if __name__ == "__main__":
    main()