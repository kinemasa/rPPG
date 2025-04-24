import cv2
import numpy as np
import math

def extract_green_channel(img_path):
    img = cv2.imread(img_path)
    return img[:, :, 1]

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    radians = math.radians(angle)
    sin = abs(math.sin(radians))
    cos = abs(math.cos(radians))
    bound_w = int(height * sin + width * cos)
    bound_h = int(height * cos + width * sin)

    M[0, 2] += (bound_w / 2) - center[0]
    M[1, 2] += (bound_h / 2) - center[1]
    return cv2.warpAffine(image, M, (bound_w, bound_h))
