import cv2
import os
import random
import matplotlib.pyplot as plt
from ml_matching.inference import extract_features

def test_random_img(data_dir, model):
  all_files = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith('.jpg')]
  img_path = random.choice(all_files)

  img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  img1 = cv2.resize(img1, (300, 300))

  rows, cols = img1.shape

  M = cv2.getRotationMatrix2D((cols/2, rows/2), 20, 1)

  img2 = cv2.warpAffine(img1, M, (cols, rows))

  kp1, desc1 = extract_features(img1, model)
  kp2, desc2 = extract_features(img2, model)

  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

  if desc1 is not None and desc2 is not None:
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches)
    plt.title(f"Top 30 matches")
    plt.axis('off')
    plt.show()
  else:
    print("Not enough matches")