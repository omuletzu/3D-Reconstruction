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

def extract_patches(img):
    patch_size = config.PATCH_SIZE
    half_p = patch_size / 2.0

    sift = cv2.SIFT_create(
        nfeatures=20000, 
        contrastThreshold=0.05
    )

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

def compute_descriptors_for_patches(patches, model):
  if len(patches) == 0:
    return None
   
  return model.predict(patches, verbose=0)

def match_images(desc1, kp1, desc2, kp2):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.array([]), np.array([]), [], []

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

    pts1_np = np.float32(pts1)
    pts2_np = np.float32(pts2)

    if len(pts1_np) >= 8:
        F, mask = cv2.findFundamentalMat(pts1_np, pts2_np, cv2.FM_RANSAC, 1.75, 0.999)

        if mask is not None:
            mask = mask.ravel() == 1
            pts1_np = pts1_np[mask]
            pts2_np = pts2_np[mask]
            
            indices1 = [indices1[i] for i in range(len(indices1)) if mask[i]]
            indices2 = [indices2[i] for i in range(len(indices2)) if mask[i]]
        else:
            return np.array([]), np.array([]), [], []
    else:
        return np.array([]), np.array([]), [], []

    return pts1_np, pts2_np, indices1, indices2