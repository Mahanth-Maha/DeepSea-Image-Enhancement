import os
import cv2

import numpy as np 
from PIL import Image

def apply_clahe(path, alpha, tile):
    img_bgr = cv2.imread(path)

    if img_bgr is None:
        print("Error: Could not load image.")
    else:
        # 1. Convert BGR to LAB color space
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        # 2. Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(img_lab)
        histL = cv2.calcHist([l], [0], None, [256], [0, 256])
        # 3. Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=tile)
        cl = clahe.apply(l)
        histCL = cv2.calcHist([cl], [0], None, [256], [0, 256])
        # 4. Merge the channels and convert back to BGR
        merged_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return enhanced_img, histL, histCL


def apply_gamma_correction(image, gamma=None):
    # gray = to_grayscale(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256])

    if gamma is None:
        mean_intensity = np.mean(gray) / 255.0
        gamma = np.log(0.5) / np.log(mean_intensity + 1e-8)
        gamma = np.clip(gamma, 0.5, 2.0)

    corrected = np.power(image / 255.0, gamma).astype(np.float32)
    corrected = np.uint8(np.clip(corrected * 255, 0, 255))

    gray_after = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    hist_after = cv2.calcHist([gray_after], [0], None, [256], [0, 256])

    return corrected, hist_before, hist_after, gamma
