import config
import os
import cv2
import numpy as np
from hardware.camera import Camera
from vision2d.processing import fix_distorsion, remove_background, enhance_contrast, remove_background_ai
# from ml_matching.inference import extract_patches, compute_descriptors_for_patches, load_model

def test():
    descriptors_arr = []

    i = 1

    path = os.path.join('data', 'dino_ring_sparse_dataset', f'dinoSR0010.png')

    image = cv2.imread(path)

    if image is None:
        print('Null image')

    cv2.imshow(f'img{i}', image)
    cv2.waitKey(0)

    image = fix_distorsion(image)

    image = remove_background_ai(image)

    image = enhance_contrast(image)

    # keypoints, patches = extract_patches(image)

    # model = load_model(config.MODEL_PATH)

    # descriptors = compute_descriptors_for_patches(patches, model)

    # descriptors_arr.append(descriptors)

    cv2.imshow(f'img{i}', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    test()