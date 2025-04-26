import cv2
import mediapipe as mp
import numpy as np

import cv2
import mediapipe as mp
import numpy as np

class FaceMaskExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

    def extract_mask(self, image):
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            points = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark], dtype=np.int32)

            # 輪郭用のインデックス（0～467すべて使用）
            hull = cv2.convexHull(points)

            # マスク作成
            mask = np.zeros_like(image)
            cv2.fillConvexPoly(mask, hull, (255, 255, 255))

            # 元画像と合成して顔のみ抽出
            masked_face = cv2.bitwise_and(image, mask)
            return masked_face
        else:
            return np.zeros_like(image)