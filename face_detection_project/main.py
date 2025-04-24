import cv2
import mediapipe as mp
import os
import glob

# Mediapipe の顔検出初期化
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 固定サイズのROI（幅, 高さ）
ROI_SIZE = (200, 200)

# 入力フォルダと出力フォルダのパス
input_folder = "c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-dataset-test\\subject1\\"
output_folder = input_folder +"face_ROIs-fixed\\"
os.makedirs(output_folder, exist_ok=True)

# 顔検出処理
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    for img_path in glob.glob(os.path.join(input_folder, "*.png")):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}")
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                # 顔の中心座標
                cx = int((bboxC.xmin + bboxC.width / 2) * w)
                cy = int((bboxC.ymin + bboxC.height / 2) * h)

                roi_w, roi_h = ROI_SIZE
                x1 = max(cx - roi_w // 2, 0)
                y1 = max(cy - roi_h // 2, 0)
                x2 = min(x1 + roi_w, w)
                y2 = min(y1 + roi_h, h)

                # 切り出し（画像の端にかかった場合はリサイズして補正）
                roi = image[y1:y2, x1:x2]
                if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
                    roi = cv2.resize(roi, ROI_SIZE)

                # 保存
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)
                out_path = os.path.join(output_folder, f"{name}_face{i}{ext}")
                cv2.imwrite(out_path, roi)
        else:
            print(f"No face detected in {img_path}")