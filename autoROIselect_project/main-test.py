import cv2
from tkinter import Tk, filedialog
from face_detector.face_detector import Param, FaceDetector
import numpy as np


# ---- ステップ1：画像ファイルを選択 ----
Tk().withdraw()  # Tkウィンドウを表示させない
image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
if not image_path:
    print("画像が選択されませんでした。")
    exit()

# ---- ステップ2：画像読み込み ----
img = cv2.imread(image_path)
if img is None:
    print("画像を読み込めませんでした。")
    exit()

# ---- ステップ3：ROI検出・描画クラスの初期化 ----
detector = FaceDetector(Param)

# ---- ステップ4：描画するROI名を指定（例：glabella）----
roi_name = "glabella"
roi_names = ["glabella", "left malar", "right malar"]
if roi_name not in Param.list_roi_name:
    print(f"{roi_name} は登録されていないROI名です。")
    exit()

# ---- ステップ5：ROI描画 ----
# img_with_roi = detector.faceMeshDraw(img.copy(), roi_name)
img_with_rois = detector.faceMeshDrawMultiple(img.copy(), roi_names)
# cv2.imshow("Multiple ROIs", img_with_rois)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # ---- ステップ6：表示・保存（任意） ----
cv2.imshow(f"ROI: {roi_name}", img_with_rois)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像を保存したい場合は以下を有効化
# cv2.imwrite("output_with_roi.png", img_with_roi)
