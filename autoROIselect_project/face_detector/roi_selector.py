
import cv2
import mediapipe as mp

class LandmarkROILocator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        # self.landmark_indices = {
        #     'forehead_center': 151,
        #     'forehead_upper': 10,
        #     'forehead_lower': 67,
        #     'forehead_left': 70,
        #     'forehead_right': 203,
        #     'forehead_top': 9,  # optional,髪の生え際に近い
        #     # その他
        #     'left_cheek': 50,
        #     'right_cheek': 280
        # }
        
        
        # self.landmark_indices = {f'point_{i}': i for i in range(468) if i not in excluded}     
        self.landmark_indices = {f'point_{i}': i for i in range(468)}
        self.initial_roi_size = None 

    def get_landmark_positions(self, image):
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        landmark_dicts = []

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                coords = {}
                for name, idx in self.landmark_indices.items():
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    coords[name] = (x, y)
                landmark_dicts.append(coords)
        return landmark_dicts

    def draw_landmarks(self, image, landmarks):
        for name, (x, y) in landmarks.items():
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            # cv2.putText(image, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image
    
    def get_roi_crops(self, image, landmarks, roi_size=40):
        h, w, _ = image.shape
        crops = {}
        for name, (x, y) in landmarks.items():
            x1 = max(0, x - roi_size // 2)
            y1 = max(0, y - roi_size // 2)
            x2 = min(w, x1 + roi_size)
            y2 = min(h, y1 + roi_size)
            roi = image[y1:y2, x1:x2]
            if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
                roi = cv2.resize(roi, (roi_size, roi_size))
            crops[name] = roi
        return crops
    
    def draw_roi_boxes(self, image, landmarks, roi_size):
        """
        ランドマークに基づき、指定された固定サイズでROIを描画する。
        roi_size: ROIの1辺の長さ（ピクセル）で固定
        """
        h, w, _ = image.shape
        half_size = roi_size // 2

        for name, (x, y) in landmarks.items():
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(w, x + half_size)
            y2 = min(h, y + half_size)
             # ROIが画像からはみ出す場合はスキップ
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # ROIの枠を描画（青）
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # ラベルも表示
            # cv2.putText(image, f"{name} ({roi_size}px)", (x1, y1 - 5),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return image

    
    def get_landmarks_and_bbox(self, image):
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 顔検出（bounding box 取得）
        detection_results = self.face_detection.process(rgb)
        # ランドマーク取得
        mesh_results = self.face_mesh.process(rgb)

        outputs = []
        if detection_results.detections and mesh_results.multi_face_landmarks:
            for detection, landmarks in zip(detection_results.detections, mesh_results.multi_face_landmarks):
                bbox = detection.location_data.relative_bounding_box
                face_bbox = {
                    'x': int(bbox.xmin * w),
                    'y': int(bbox.ymin * h),
                    'width': int(bbox.width * w),
                    'height': int(bbox.height * h)
                }
                coords = {}
                for name, idx in self.landmark_indices.items():
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    coords[name] = (x, y)
                outputs.append((coords, face_bbox))
        return outputs
    
    def draw_face_bbox(self, image, face_bbox, color=(0, 255, 255), thickness=2):
        """
        顔の検出範囲（bounding box）を画像上に表示する
        """
        x1 = face_bbox['x']
        y1 = face_bbox['y']
        x2 = x1 + face_bbox['width']
        y2 = y1 + face_bbox['height']
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        # cv2.putText(image, "Face BBox", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image
