import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


## 自作ライブラリ
from mycommon.select_folder import select_folder
from mycommon.road_and_save import get_sorted_image_files,get_sorted_image_100files,save_pulse_to_csv
from mycommon.extract_pulsewave import select_roi,compute_g_means
from mycommon.visualize_pulsewave import plot_pulse_wave
from mycommon.processing_pulse import detrend_pulse,sg_filter_pulse,bandpass_filter_pulse
from autoROIselect_project.face_detector.face_detector import FaceDetector, Param
from mycommon.select_folder import select_folder,select_file
from autoROIselect_project.face_detector.face_detector import FaceDetector, Param

def main():
    # ROI名の指定（必要に応じて変更可能）
    roi_names = ['left malar', 'right malar', 'glabella']

    # フォルダ選択
    folder = select_folder()
    if not folder:
        print("フォルダが選択されませんでした。")
        return

    image_files = get_sorted_image_files(folder,600)
    if not image_files:
        print("指定フォルダに画像ファイルが見つかりません。")
        return

    # FaceDetector 初期化
    params = Param()
    detector = FaceDetector(params)

    print("画像群に対するROI描画を開始します。 'q' で途中終了可能です。")

    for path in image_files:
        img = cv2.imread(path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {path}")
            continue

        # # ROI描画
        img_with_roi = detector.faceMeshDrawMultiple(img, roi_names)

        # 表示
        cv2.imshow("ROI Tracking on Images", img_with_roi)
        key = cv2.waitKey(16)  # 300msごとに切り替え
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()