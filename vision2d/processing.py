import cv2
import json
import numpy as np
from rembg import remove

def resize_image(image, target_width):
    if image is None:
        return None

    original_height, original_width = image.shape[:2]

    scale_factor = target_width / original_width
    target_height = int(original_height * scale_factor)

    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return resized_image

# def fix_distorsion(image, K, D):
#     h, w = image.shape[:2]

#     new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

#     dst = cv2.undistort(image, K, D, None, new_camera_mtx)

#     x, y, w_roi, h_roi = roi
#     dst = dst[y : y + h_roi, x : x + w_roi]

#     final_K = new_camera_mtx.copy()
#     final_K[0, 2] -= x
#     final_K[1, 2] -= y

#     return dst, final_K

def fix_distorsion(image, K, D):
    # Aplicăm corecția de distorsiune direct, păstrând dimensiunea 
    # și matricea K exact așa cum le-a calculat calibrarea.
    dst = cv2.undistort(image, K, D)
    
    # Returnăm imaginea corectată și matricea K originală, nealterată
    return dst, K

def enhance_contrast(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    enhanced_image = clahe.apply(gray)

    return enhanced_image

def remove_background(image):
    h, w = image.shape[:2]

    margin_x = int(w * 0.02)
    margin_y = int(h * 0.02)

    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    result = cv2.bitwise_and(image, image, mask=binary_mask)

    cv2.imshow('img', result)
    cv2.waitKey(0)

    return result

def remove_background_ai(image):

    result_rgba = remove(image)

    image_bgr = cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGR)

    ai_mask = result_rgba[:, :, 3]

    return image_bgr, ai_mask