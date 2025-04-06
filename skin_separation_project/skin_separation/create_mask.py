import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import mediapipe as mp  
import numpy as np
from scipy.optimize import minimize
# MediapipeのFaceDetectionを初期化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ================================================================#
# 顔マスク用関数　#
# ================================================================#
def create_black_mask(image_rgb):
    """
    黒色部分以外を1、黒色部分を0とするマスクを生成する関数
    """
    # 各ピクセルが完全な黒かどうかを判定（全てのチャネルが0）
    mask = np.all(image_rgb == [0, 0, 0], axis=-1)
    # 黒色以外の部分を1に、黒色部分を0にする
    inverse_mask = np.logical_not(mask).astype(np.float32)
    return inverse_mask


def create_face_mask(image_rgb,erosion_size=5):
    """
    Mediapipeを使って顔と耳の領域を含むマスクを作成する関数
    """
    image_height, image_width = image_rgb.shape[:2]
    face_mask = np.zeros((image_height, image_width), dtype=np.float32)  # マスク初期化

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 顔のランドマークを2D座標に変換し、輪郭を描画
                points = [(int(landmark.x * image_width), int(landmark.y * image_height))
                          for landmark in face_landmarks.landmark]

                # 輪郭をポリゴンで描画
                convex_hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(face_mask, convex_hull, 1)
                
        # 収縮処理でマスクを輪郭より内側に調整
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    inner_face_mask = cv2.erode(face_mask, kernel, iterations=1)

    return inner_face_mask

def create_hsv_mask(image_rgb, hue_range=(0, 180), saturation_range=(0, 255), value_range=(0, 255)):
    """
    HSV色空間で特定の範囲に基づいたマスクを作成する関数。

    Parameters:
        image_rgb (numpy.ndarray): RGB画像。
        hue_range (tuple): 色相(H)の範囲 (0-180)。
        saturation_range (tuple): 彩度(S)の範囲 (0-255)。
        value_range (tuple): 明度(V)の範囲 (0-255)。

    Returns:
        mask (numpy.ndarray): HSVマスク（0または1の値を持つ2D配列）。
    """
    # RGB画像をBGRに変換
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # BGR画像をHSV色空間に変換
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 範囲指定でマスクを作成
    lower_bound = np.array([hue_range[0], saturation_range[0], value_range[0]])
    upper_bound = np.array([hue_range[1], saturation_range[1], value_range[1]])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # マスクを二値化して0または1にする
    mask = (mask > 0).astype(np.float32)
    return mask

def create_eye_mask(image):
    """
    入力された画像の目の部分を検出し、その領域を黒く（値を0）し、その他の領域を1にする関数。

    Args:
        image (numpy.ndarray): 入力画像（RGB形式）。

    Returns:
        numpy.ndarray: 目の領域が黒くマスクされ、その他の領域が1に設定された画像。
    """
    # MediapipeのFace Meshソリューションを初期化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Mediapipeで顔ランドマークを検出
    results = face_mesh.process(image)

    # 元の画像と同じサイズのマスクを作成
    mask = np.ones(image.shape[:2], dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 目のランドマークインデックス
            left_eye_indices = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]
            right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374]

            # 左目と右目のランドマーク座標を取得
            h, w, _ = image.shape
            left_eye_points = [(int(landmark.x * w), int(landmark.y * h)) for i, landmark in enumerate(face_landmarks.landmark) if i in left_eye_indices]
            right_eye_points = [(int(landmark.x * w), int(landmark.y * h)) for i, landmark in enumerate(face_landmarks.landmark) if i in right_eye_indices]

            # 左目全体と右目全体のポリゴンを閉じた形で作成
            left_eye_hull = cv2.convexHull(np.array(left_eye_points, dtype=np.int32))
            right_eye_hull = cv2.convexHull(np.array(right_eye_points, dtype=np.int32))

            # ポリゴンとして目の領域をマスク
            cv2.fillPoly(mask, [left_eye_hull], 0)
            cv2.fillPoly(mask, [right_eye_hull], 0)
    # 目の領域を0、その他を1として適用
    result_image = mask

    # リソースを解放
    face_mesh.close()

    return result_image