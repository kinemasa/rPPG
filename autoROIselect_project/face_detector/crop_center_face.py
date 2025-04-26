"""
顔画像の中心を検出し、指定したroiの大きさで画像を切り抜く
"""

import cv2
import mediapipe as mp

class FaceCenterCropper:
    def __init__(self, roi_size=(224, 224)):
        self.roi_size = roi_size
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

    def crop(self, image):
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        crops = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                cx = x + width // 2
                cy = y + height // 2

                roi_w, roi_h = self.roi_size
                x1 = max(0, cx - roi_w // 2)
                y1 = max(0, cy - roi_h // 2)
                x2 = min(w, x1 + roi_w)
                y2 = min(h, y1 + roi_h)

                roi = image[y1:y2, x1:x2]
                if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
                    roi = cv2.resize(roi, self.roi_size)

                crops.append(roi)
        return crops