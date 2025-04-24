from face_detector.crop_center_face import FaceCenterCropper
from face_detector.extract_face_mask import FaceMaskExtractor
import cv2
import os
image = cv2.imread("c:\\Users\\kine0\\tumuraLabo\\programs\\rPPG\\rPPG\\UBFC-dataset-test\\subject1\\frame_00000.png")

import sys
# utils.py までのパスを通す（相対的に）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','mycommon')))
from myutils import select_folder


folder = select_folder("")

# ROI切り出し
cropper = FaceCenterCropper()
cropped_faces = cropper.crop(image)

# 顔マスク抽出
masker = FaceMaskExtractor()
masked_face = masker.extract_mask(image)

cv2.imshow("Cropped", cropped_faces[0])
cv2.imshow("Masked", masked_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
