import cv2
import json
import numpy as np
from rembg import remove

with open('data/camera_params.json', 'r') as f:
    params = json.load(f)
    K_mtx = np.array(params['camera_matrix'])
    D_params = np.array(params['distortion_params'])

def fix_distorsion(image):
    return image
    
    # h, w = image.shape[:2]

    # new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K_mtx, D_params, (w, h), 1, (w, h))

    # dst = cv2.undistort(image, K_mtx, D_params, None, new_camera_mtx)

    # x, y, w, h = roi

    # dst = dst[y : y + h, x : x + w]

    # return dst

def enhance_contrast(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
    result = remove(image)

    return result