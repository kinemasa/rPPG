from face_detector.crop_center_face import FaceCenterCropper
from face_detector.extract_face_mask import FaceMaskExtractor
import cv2
import os
from face_detector.roi_selector import LandmarkROILocator
image = cv2.imread("c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-dataset-test\\subject1\\frame_00000.png")
cap = cv2.VideoCapture("c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-data1.avi")
locator = LandmarkROILocator()
import sys
# utils.py までのパスを通す（相対的に）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','mycommon')))

while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        roi_size = None
        # ランドマーク＋顔のバウンディングボックス（顔サイズ）取得
        landmark_and_bbox_list = locator.get_landmarks_and_bbox(frame)

        for landmarks, bbox in landmark_and_bbox_list:
            frame = locator.draw_face_bbox(frame, bbox) 
            if roi_size is None:
                roi_size = int(bbox['width'] * 0.1)  # 初回だけ固定サイズを決定
            
            # 顔サイズに基づいたROIボックス表示
            frame = locator.draw_roi_boxes(frame, landmarks,roi_size)

            # ランドマーク（点）も描画
            # frame = locator.draw_landmarks(frame, landmarks)

        # 表示
        cv2.imshow("Landmark and Dynamic ROI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# # ROI切り出し
# cropper = FaceCenterCropper()
# cropped_faces = cropper.crop(image)

# # 顔マスク抽出
# masker = FaceMaskExtractor()
# masked_face = masker.extract_mask(image)

# cv2.imshow("Cropped", cropped_faces[0])
# cv2.imshow("Masked", masked_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     landmark_list = locator.get_landmark_positions(frame)

#     for landmarks in landmark_list:
#         frame = locator.draw_landmarks(frame, landmarks)

#     cv2.imshow("ROI landmarks", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     landmark_list = locator.get_landmark_positions(frame)
#     for landmarks in landmark_list:
#         frame = locator.draw_roi_boxes(frame, landmarks, roi_size=40)  # ←ここが追加部分
#         frame = locator.draw_landmarks(frame, landmarks)  # ランドマーク点も同時に表示

#     cv2.imshow("ROI Boxes and Landmarks", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
