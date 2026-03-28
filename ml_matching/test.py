import cv2
import os
import random
import config
import matplotlib.pyplot as plt
from ml_matching.inference import extract_patches, compute_descriptors_for_patches
from ml_matching.inference import load_model

def test_random_img(data_dir, model):
    all_files = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith('.jpg')]
    img_path = random.choice(all_files)

    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (300, 300))

    rows, cols = img1.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), 20, 1)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    kp1, p1 = extract_patches(img1)
    kp2, p2 = extract_patches(img2)

    desc1 = compute_descriptors_for_patches(p1, model)
    desc2 = compute_descriptors_for_patches(p2, model)

    if desc1 is not None and desc2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        knn_matches = bf.knnMatch(desc1, desc2, k=2)

        good_matches = []
        ratio_thresh = 0.8
        
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        print(f"Model found {len(good_matches)} matches")
        
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        img_matches = cv2.drawMatches(
            img1, kp1, 
            img2, kp2, 
            good_matches[:50], 
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)
        )

        plt.figure(figsize=(15, 10))
        plt.imshow(img_matches)
        plt.title(f"Test Model ML: Rotație 20° | {len(good_matches)} Good Matches")
        plt.axis('off')
        plt.show()
    else:
        print("Cannot extract.")

if __name__ == "__main__":
    import tensorflow as tf
    
    model = load_model(config.MODEL_PATH)
    
    data_directory = 'data/kendama_ring_dataset'
    
    test_random_img(data_directory, model)