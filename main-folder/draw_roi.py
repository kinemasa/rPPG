import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## 自作ライブラリ
from mycommon.select_folder import select_folder,select_file
from autoROIselect_project.face_detector.face_detector import FaceDetector, Param

def main():
    # 画像ファイル選択
    img_path = select_file()
    if not img_path:
        print("画像ファイルが選択されませんでした。")
        return

    # 画像読み込み
    img = cv2.imread(img_path)
    if img is None:
        print("画像の読み込みに失敗しました。パスを確認してください。")
        return

    # ROIを選択（複数可）
    roi_names = ["lower medial forehead","glabella","left lower lateral forehead","right lower lateral forehead","upper nasal dorsum","left malar","right malar","left lower cheek","right lower cheek","chin"]  # 必要に応じて変更

    # 顔検出とROI描画
    params = Param()
    detector = FaceDetector(params)
    img_with_roi = detector.faceMeshDrawMultiple(img, roi_names)

    # 結果を表示
    # cv2.imshow("ROI Overlay", img_with_roi)
    cv2.imwrite("result.png",img_with_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
