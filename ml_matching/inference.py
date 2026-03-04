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

def extract_patches(img, model):
  patch_size = 32
  half_p = patch_size // 2

  fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
  keypoints = fast.detect(img, None)

  valid_keypoints = []
  patches = []

  h, w = img.shape

  for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])

    if x < half_p or y < half_p or x >= w - half_p or y >= h - half_p:
      continue

    patch = img[y - half_p : y + half_p, x - half_p : x + half_p]

    if patch.shape != (patch_size, patch_size):
      continue

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
    MAX_MATCHES = 1000

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    pts1 = []
    pts2 = []

    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.array([]), np.array([])

    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    matches = matches[:MAX_MATCHES]

    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        pt1 = kp1[idx1].pt
        pt2 = kp2[idx2].pt

        pts1.append(pt1)
        pts2.append(pt2)

    pts1_np = np.float32(pts1)
    pts2_np = np.float32(pts2)

    return pts1_np, pts2_np