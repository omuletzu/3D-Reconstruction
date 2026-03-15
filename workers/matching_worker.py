import config
import cv2
from ml_matching.inference import compute_descriptors_for_patches, match_images

def matching_worker(model, matching_queue, processed_data, lock, all_matches, geometry_queue):

    sift = cv2.SIFT_create()

    while True:
        idx = matching_queue.get()

        if idx is None:
            break

        print(f"[ML_THREAD] inferring for image {idx}")

        # patches = processed_data[idx]['patches']

        # descriptors = compute_descriptors_for_patches(patches, model)

        img = processed_data[idx]['image_gray']
        keypoints = processed_data[idx]['keypoints']

        _, descriptors = sift.compute(img, keypoints)

        with lock:
            processed_data[idx]['descriptors'] = descriptors

        window_size = 4
        
        neighbours_to_check = []
        for i in range(-window_size, window_size + 1):
            if i == 0: continue
            neighbours_to_check.append((idx + i) % config.TOTAL_PHOTOS)

        for neighbour in neighbours_to_check:
            if idx < neighbour:
                img_A, img_B = idx, neighbour
            else:
                img_A, img_B = neighbour, idx

            pair_name = f"{img_A}_{img_B}"

            is_new_pair = False

            with lock:
                if pair_name in all_matches:
                    continue

                ready_A = (img_A in processed_data) and (processed_data[img_A].get('descriptors') is not None)
                ready_B = (img_B in processed_data) and (processed_data[img_B].get('descriptors') is not None)

                if ready_A and ready_B:
                    is_new_pair = True
                    all_matches[pair_name] = "PENDING"

            if is_new_pair:
                print(f"[MATCHING] Starting for {pair_name}")

                descA = processed_data[img_A]['descriptors']
                kpA = processed_data[img_A]['keypoints']

                descB = processed_data[img_B]['descriptors']
                kpB = processed_data[img_B]['keypoints']

                ptsA, ptsB, idA, idB = match_images(descA, kpA, descB, kpB)

                geometry_queue.put({
                    'pair_name': pair_name,
                    'ptsA': ptsA,
                    'ptsB': ptsB,
                    'indicesA': idA,
                    'indicesB': idB
                })

                print(f"[MATCHING] Done for {pair_name}")

        matching_queue.task_done()