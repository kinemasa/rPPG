# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from scipy import interpolate

# @ numba.jit
def skinSeparation(img):
    """
    Converts a BMP image to a hemoglobin component image.

    Args:
        input_image_path (str): Path to the input BMP image.

    Returns:
        np.ndarray: 16-bit hemoglobin component image.
    """

    # Define melanin and hemoglobin coefficients
    melanin = np.array([ 0.4143, 0.3570, 0.8372 ], dtype=np.float32)
    hemoglobin = np.array([ 0.2988, 0.6838, 0.6657 ], dtype=np.float32)

    image_height, image_width, _ = img.shape

    # Gamma correction parameters
    aa = 1
    bb = 0
    gamma = 1
    cc = 0
    gg = [1, 1, 1]
    DC = 1 / 255

    # Color vectors and illumination intensity vector
    vec = np.zeros((3, 3), dtype=np.float32)
    vec[0] = np.array([1.0, 1.0, 1.0])
    vec[1] = melanin
    vec[2] = hemoglobin

    # Normal vector of skin color distribution plane
    norm = [
        vec[1,1]*vec[2,2] - vec[1,2]*vec[2,1],
        vec[1,2]*vec[2,0] - vec[1,0]*vec[2,2],
        vec[1,0]*vec[2,1] - vec[1,1]*vec[2,0],
    ]

    # Convert image to linear RGB and normalize
    linearSkin = np.zeros_like(img, dtype=np.float32).transpose(2, 0, 1)
    skin = img.transpose(2, 0, 1).astype(np.float32)

    for i in range(3):
        linearSkin[i] = np.power(((skin[i] - cc) / aa), (1 / gamma) - bb) / gg[i] / 255

    # Create mask images
    img_mask = np.zeros_like(linearSkin, dtype=np.float32)
    img_mask2 = DC + np.zeros_like(linearSkin, dtype=np.float32)
    img_mask[linearSkin > 0.0] = 1  # Mask (0 or 1)
    img_mask2[linearSkin > 0.0] = 0  # Mask (DC or 0)

    # Convert to concentration space (log space)
    S = -np.log(linearSkin + img_mask2) * img_mask

    # Subtract minimum skin tone
    MinSkin = [0, 0, 0]
    for i in range(3):
        S[i] = S[i] - MinSkin[i]

    # Find intersection of illumination variation direction with skin color distribution plane
    t = -(
        norm[0] * S[0] + norm[1] * S[1] + norm[2] * S[2]
    ) / (
        norm[0] * vec[0, 0] + norm[1] * vec[0, 1] + norm[2] * vec[0, 2]
    )

    # Shadow removal
    skin_flat = (t[np.newaxis, :, :].transpose(1, 2, 0) * vec[0]).transpose(2, 0, 1) + S

    # Compute pigment concentrations
    CompExtM = np.linalg.pinv(np.vstack([melanin, hemoglobin]).transpose())
    Compornent = np.dot(CompExtM, skin_flat.reshape(3, image_height * image_width))
    Compornent = Compornent.reshape(2, image_height, image_width)

    # Extract hemoglobin component
    L_Hem = Compornent[1]

    # Scale to 16-bit range
    hemoglobin_image = np.clip(L_Hem, 0, 1) * (2**16 - 1)
    hemoglobin_image = hemoglobin_image.astype(np.uint16)

    return hemoglobin_image
