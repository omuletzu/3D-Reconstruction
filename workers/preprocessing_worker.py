import cv2
import numpy as np
from vision2d.processing import remove_background_ai, enhance_contrast, fix_distorsion
from ml_matching.inference import extract_patches

def preprocessing_worker(preprocessing_queue, lock, processed_data, matching_queue):
    while True:
        item = preprocessing_queue.get()
        if item is None: break
        path, idx = item

        image = cv2.imread(path)
        if image is None:
            preprocessing_queue.task_done()
            continue

        image_undistorted = fix_distorsion(image)

        image_contrast = enhance_contrast(image_undistorted)

        keypoints, patches = extract_patches(image_contrast)

        image_no_bg = remove_background_ai(image_undistorted)
        gray_no_bg = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
        
        _, raw_mask = cv2.threshold(gray_no_bg, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8) 
        safe_mask = cv2.erode(raw_mask, kernel, iterations=1)

        valid_keypoints = []
        valid_patches = []
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])

            if 0 <= y < safe_mask.shape[0] and 0 <= x < safe_mask.shape[1]:
                if safe_mask[y, x] > 0:
                    valid_keypoints.append(kp)
                    valid_patches.append(patches[i])

        valid_keypoints = np.array(valid_keypoints)
        valid_patches = np.array(valid_patches)

        with lock:
            processed_data[idx] = {
                'image_gray': cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY),
                'image_color': image_no_bg,
                'keypoints': valid_keypoints,
                'patches': valid_patches,
                'descriptors': None
            }

        matching_queue.put(idx)
        
        preprocessing_queue.task_done()