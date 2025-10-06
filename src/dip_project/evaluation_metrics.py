import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def structural_similarity_index(imageA, imageB):
    # Ensure same size
    if imageA.shape != imageB.shape:
        raise ValueError(f"Image dimensions do not match: {imageA.shape} vs {imageB.shape}")

    imageA = imageA.astype(np.uint8)
    imageB = imageB.astype(np.uint8)

    ssim_score, diff = ssim(
        imageA,
        imageB,
        channel_axis=-1,  
        full=True,
        data_range=255
    )
    return ssim_score, diff

# UCIQE= 0.4680σc​+0.2745μs​+0.2576σl​

import numpy as np
from skimage.color import rgb2lab, rgb2hsv

def UCIQE(image_rgb):
    """
    image_rgb: HxWx3 uint8, RGB order (0-255)
    returns: scalar UCIQE (expected ~0.2-0.8 for many images)
    """
    # normalize to [0,1]
    img = image_rgb.astype(np.float32) / 255.0

    # Lab with skimage: L in [0,100], a/b around -128..127
    lab = rgb2lab(img)
    L = lab[:, :, 0]        # 0..100
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # chroma
    chroma = np.sqrt(a**2 + b**2)  # absolute chroma
    sigma_c = np.std(chroma) / 100.0   # normalize (~0-1)
    sigma_l = np.std(L) / 100.0        # normalize (~0-1)

    # use HSV saturation in [0,1]
    hsv = rgb2hsv(img)
    mean_s = np.mean(hsv[:, :, 1])

    # constants from Yang & Sowmya (UCIQE)
    uciqe = 0.4680 * sigma_c + 0.2745 * mean_s + 0.2576 * sigma_l
    return float(uciqe)


def UICM(img):
    R, G, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rg = R - G
    yb = 0.5 * (R + G) - B
    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)
    uicm = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return uicm
def UISM(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    uism = np.mean(gradient_magnitude)
    return uism
def UIConM(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    con_m = np.std(gray)
    return con_m

def UIQM(img):
    uicm = UICM(img)
    uism = UISM(img)
    uiconm = UIConM(img)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm, uicm, uism, uiconm






