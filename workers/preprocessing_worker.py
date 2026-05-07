import cv2
import config
import time
import numpy as np
from vision2d.processing import remove_background_ai, enhance_contrast, fix_distorsion, resize_image
from ml_matching.inference import extract_patches

def preprocessing_worker(preprocessing_queue, lock, processed_data, matching_queue, K, D, thread_metrics):

    while True:
        item = preprocessing_queue.get()

        if item is None: 
            break

        path, idx = item

        preprocessing_start_timer = time.perf_counter()

        image = cv2.imread(path)

        if image is None:
            preprocessing_queue.task_done()
            continue

        print(f"[PREPROCESSING] Started for image {idx}")

        image = resize_image(image, config.PIPELINE_WIDTH)

        image_undistorted, K_final = fix_distorsion(image, K, D)

        image_no_bg, ai_mask = remove_background_ai(image_undistorted)

        image_contrast = enhance_contrast(image_no_bg)

        cv2.imwrite(f"{config.DATA_FOLDER}/{idx}.png", image_contrast)

        keypoints, patches = extract_patches(image_contrast)

        kernel = np.ones((3, 3), np.uint8) 
        safe_mask = cv2.erode(ai_mask, kernel, iterations=1)

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

        preprocessing_end_timer = time.perf_counter()

        with lock:
            thread_metrics['prep_time_sum'] += preprocessing_end_timer - preprocessing_start_timer
            thread_metrics['prep_count'] += 1

            processed_data[idx] = {
                'image_gray': image_contrast,
                'image_color': image_no_bg,
                'keypoints': valid_keypoints,
                'patches': valid_patches,
                'descriptors': None,
                'K': K_final
            }

        print(f"[PREPROCESSING] Done for image {idx}. Extracted {len(keypoints)} features.")

        matching_queue.put(idx)
        
        preprocessing_queue.task_done()