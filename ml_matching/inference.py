import config

import cv2
import numpy as np
from ml_matching.arhitecture import build_model

def load_model(model_path):
  model = build_model(input_shape=(32, 32, 1))

  try:
      model.load_weights(model_path)
      print("Succesfully loaded weigths")
  except Exception as e:
      print(f"Couldn't load weights: {e}")

  return model

def extract_features_adaptive(img, min_features=1000, max_features=10000):
    if img is None:
        return None, None

    sift = cv2.SIFT_create(contrastThreshold=0.04)
    kp, desc = sift.detectAndCompute(img, None)

    if len(kp) < min_features:

        sift_sensitive = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=15)

        kp, desc = sift_sensitive.detectAndCompute(img, None)

    if len(kp) > max_features:

        combined = list(zip(kp, desc))

        combined.sort(key=lambda x: x[0].response, reverse=True)

        combined = combined[:max_features]

        kp, desc = zip(*combined)
        kp = list(kp)
        desc = np.array(desc)

    return kp, desc

def extract_patches(img):
    patch_size = config.PATCH_SIZE
    half_p = patch_size / 2.0

    sift = cv2.SIFT_create(nfeatures=50000, contrastThreshold=0.005, edgeThreshold=20)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # keypoints, descriptors = extract_features_adaptive(img)

    keypoints = sift.detect(img, None)

    valid_keypoints = []
    patches = []

    h, w = img.shape

    for kp in keypoints:
        x, y = kp.pt
        angle = kp.angle
        
        diag = int(np.ceil(patch_size * 1.414 / 2))
        if x < diag or y < diag or x >= w - diag or y >= h - diag:
            continue

        M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        
        M[0, 2] += half_p - x
        M[1, 2] += half_p - y

        patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=cv2.INTER_LINEAR)

        patch = patch.astype(np.float32) / 255.0
        patch = np.expand_dims(patch, axis=-1)

        patches.append(patch)
        valid_keypoints.append(kp)

    return valid_keypoints, np.array(patches)

    # return valid_keypoints, np.array(patches), descriptors

def compute_descriptors_for_patches(patches, model):
  if len(patches) == 0:
    return None
   
  return model.predict(patches, verbose=0)

def match_images(desc1, kp1, desc2, kp2):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.array([]), np.array([]), [], []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    pts1 = []
    pts2 = []
    indices1 = []
    indices2 = []

    for m, n in raw_matches:
        if m.distance < 0.8 * n.distance:
            idx1 = m.queryIdx
            idx2 = m.trainIdx

            pts1.append(kp1[idx1].pt)
            pts2.append(kp2[idx2].pt)

            indices1.append(idx1)
            indices2.append(idx2)

    return np.float32(pts1), np.float32(pts2), indices1, indices2