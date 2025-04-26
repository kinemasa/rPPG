
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## 自作ライブラリ
from skin_separation_project.color_vector_estimation.estimator import estimate_color_vectors_from_files

def main():
    print("メラニン・ヘモグロビンベクトルの推定を開始します（ファイル選択）")
    melanin, hemoglobin = estimate_color_vectors_from_files()

    print("\n===== 推定結果 =====")
    print(f"melanin    = {melanin}")
    print(f"hemoglobin = {hemoglobin}")
    print("====================")
    print("推定完了")

if __name__ == "__main__":
    main()