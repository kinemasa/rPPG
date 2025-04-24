import cv2
import numpy as np
import os

def integrate_all(images, output_path, filename="addmeanAll.png"):
    if len(images) == 0:
        raise ValueError("No images provided")

    img_sample = cv2.imread(images[0], 0)
    h, w = img_sample.shape
    acc = np.zeros((h, w), dtype=np.float32)

    for path in images:
        img = cv2.imread(path, 0)
        acc += img

    acc /= len(images)
    acc = acc.astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, filename), acc)

def integrate_in_groups(images, group_size, output_path):
    if len(images) < group_size:
        raise ValueError("Not enough images to form a group")

    img_sample = cv2.imread(images[0], 0)
    h, w = img_sample.shape

    for i in range(0, len(images) - group_size + 1, group_size):
        acc = np.zeros((h, w), dtype=np.float32)
        for j in range(group_size):
            img = cv2.imread(images[i + j], 0)
            acc += img

        acc /= group_size
        acc = acc.astype(np.uint8)
        out_file = os.path.join(output_path, f"group_{i}.png")
        cv2.imwrite(out_file, acc)
