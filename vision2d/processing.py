import cv2
import json
import numpy as np

with open('data/camera_params.json', 'r') as f:
    params = json.load(f)
    K_mtx = np.array(params['camera_matrix'])
    D_params = np.array(params['distortion_params'])

def fix_distorsion(image):
    h, w = image.shape[:2]

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K_mtx, D_params, (w, h), 1, (w, h))

    dst = cv2.undistort(image, K_mtx, D_params, None, new_camera_mtx)

    x, y, w, h = roi

    dst = dst[y : y + h, x : x + w]

    return dst

def enhance_contrast(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced_image = clahe.apply(gray)

    return enhanced_image

def remove_background(image, reference_bg, threshold):
    diff = cv2.absdiff(image, reference_bg)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result