from common.imports import os,pd,joblib


def save_subject_results_to_csv(subject_results):
    """
    被験者ごとの推定結果をCSVファイルに保存する

    Parameters
    ----------
    subject_results : list
        各被験者の結果を格納したリスト
    """
    output_dir = "subject_results\\"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"subject_results.csv")
    
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

def save_best_params(best_params):
    OUTPUT_FILE = "best_params.txt"
    with open(OUTPUT_FILE, "a") as f:  # 追記モードで開く
        for model in best_params:
            f.write(f"{model}:\n")
            for param in best_params[model]:
                f.write(f"  {param}: {best_params[model][param]}\n")
            f.write("\n")

def save_selected_features(model_name, target, selected_features):
    OUTPUT_FILE = "selected_features.txt"
    with open(OUTPUT_FILE, "a") as f:  # ファイルに追記
        f.write(f"Model: {model_name}, Target: {target}\n")
        f.write(f"Selected Features: {selected_features}\n\n")

def save_model(model, model_name, target, select):
    MODEL_SAVE_DIR = "saved_models\\"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    file_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{target}_{select}.pkl")
    joblib.dump(model, file_path)
