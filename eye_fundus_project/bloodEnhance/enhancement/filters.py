import cv2
import numpy as np
from skimage.exposure import rescale_intensity

def convolve2D(image, kernel):
    iH, iW = image.shape
    kH, kW = kernel.shape
    pad = (kW - 1) // 2
    img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    for y in range(pad, iH + pad):
        for x in range(pad, iW + pad):
            roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            output[y - pad, x - pad] = (roi * kernel).sum()

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    residual = image - output
    return output, residual

def isotropic_wavelet_filter(image):
    c_prev = image
    C1, C2, C3 = 1./16., 4./16., 6./16.
    kernel_sizes = [5, 9, 17]
    W = []

    for ks in kernel_sizes:
        half_ks = ks // 2
        kernel = np.zeros((1, ks), dtype='float32')
        kernel[0][0] = kernel[0][-1] = C1
        kernel[0][half_ks//2] = kernel[0][half_ks + half_ks//2] = C2
        kernel[0][half_ks] = C3

        c_next, w = convolve2D(c_prev, kernel.T @ kernel)
        c_prev = c_next
        W.append(w)

    wavelet_enhanced = cv2.medianBlur(W[1], 3) + cv2.medianBlur(W[2], 3)
    return wavelet_enhanced