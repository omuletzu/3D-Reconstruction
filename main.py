import config

import os
import time
import threading
import queue
import cv2
import json
import numpy as np
from hardware.camera import Camera
from hardware.stepper import ArduinoController
from vision2d.processing import fix_distorsion, remove_background, enhance_contrast
from ml_matching.inference import extract_patches, compute_descriptors_for_patches, match_images, load_model
from geometry3d.ransac import ransac
from geometry3d.utils import extract_extrinsics_E, get_best_solution, triangulate_points, save_point_cloud_ply

processing_queue = queue.Queue()

ml_queue = queue.Queue()

processed_data = {}

all_matches = {}

lock = threading.Lock()

def thread_worker():
    while True:
        item = processing_queue.get()

        if item is None:
            break

        path, idx = item

        print(f"[THREAD] Processing image {idx} - {path}")

        image = cv2.imread(path)

        if image is None:
            processing_queue.task_done()
            continue

        image = fix_distorsion(image)

        image = remove_background(image)

        image = enhance_contrast(image)

        keypoints, patches = extract_patches(image)

        with lock:
            processed_data[idx] = {
                'keypoints': keypoints,
                'patches': patches,
                'descriptors': None
            }

        ml_queue.put(idx)

        print(f"[THREAD] Processed image {idx}")
        
        processing_queue.task_done()

def ml_and_matching_worker(model, K):
    while True:
        idx = ml_queue.get()

        if idx is None:
            break

        print(f"[ML THREAD] infering for image {idx}")

        patches = processed_data[idx]['patches']

        descriptors = compute_descriptors_for_patches(patches, model)

        with lock:
            processed_data[idx]['descriptors'] = descriptors

        neighbours_to_check = [
            (idx - 2) % config.TOTAL_PHOTOS,
            (idx - 1) % config.TOTAL_PHOTOS,
            (idx + 1) % config.TOTAL_PHOTOS,
            (idx + 2) % config.TOTAL_PHOTOS
        ]

        for neighbour in neighbours_to_check:
            if neighbour == (idx + 1) % config.TOTAL_PHOTOS or neighbour == (idx + 2) % config.TOTAL_PHOTOS:
                img_A, img_B = idx, neighbour
            else:
                img_A, img_B = neighbour, idx

            pair_name = f"{img_A}_{img_B}"

            with lock:
                if pair_name in all_matches:
                    continue

                ready_A = (img_A in processed_data) and (processed_data[img_A].get('descriptors') is not None)

                ready_B = (img_B in processed_data) and (processed_data[img_B].get('descriptors') is not None)

                ready = ready_A and ready_B

            if ready:
                print(f"[MATCHING] Starting for {pair_name}")

                descA = processed_data[img_A]['descriptors']
                kpA = processed_data[img_A]['keypoints']

                descB = processed_data[img_B]['descriptors']
                kpB = processed_data[img_B]['keypoints']

                ptsA, ptsB = match_images(descA, kpA, descB, kpB)

                print(f"[GEOMETRY] Ransac for {pair_name}")

                E, ptsA_filt, ptsB_filt = ransac(ptsA, ptsB, K)

                solutions = extract_extrinsics_E(E)

                R_local, t_local = get_best_solution(solutions, ptsA_filt, ptsB_filt, K)

                with lock:
                    all_matches[pair_name] = {
                        'ptsA': ptsA_filt,
                        'ptsB': ptsB_filt,
                        'R_local': R_local,
                        't_local': t_local
                    }

                print(f"[MATCHING] Done for {pair_name}")

        ml_queue.task_done()

def main():
    if not os.path.exists(config.DATA_FOLDER):
        os.makedirs(config.DATA_FOLDER)

    print("Hardware initialization")

    arduino = ArduinoController(port=config.ARDUINO_PORT)
    camera = Camera(camera_index=config.CAMERA_INDEX)

    if not os.path.exists(config.CALIBRATION_FILE):
        camera.calibrate_camera()

    with open(config.CALIBRATION_FILE, 'r') as f:
        cam_data = json.load(f)

        K = np.array(cam_data['K'])

    model = load_model(config.MODEL_PATH)

    threads = []

    for _ in range(config.THREADS_NR):
        t = threading.Thread(target=thread_worker)

        t.start()

        threads.append(t)

    ml_threads = []

    for _ in range(config.ML_THREADS_NR):
        t = threading.Thread(target=ml_and_matching_worker, args=(model, K))

        t.start()

        ml_threads.append(t)

    time.sleep(2)

    print("Started photo scanning")

    try:
        for i in range(config.TOTAL_PHOTOS):
            print(f"Capturing frame {i}")

            if arduino.rotate_step():
                time.sleep(0.5)

                image = camera.captura_frame()

                if image:
                    image_file_path = os.path.join(config.DATA_FOLDER, f"photo_{i}.jpg")

                    cv2.imwrite(image_file_path, image)

                    processing_queue.put((image_file_path, i))

                    print(f"Saved image {i}")
                else:
                    print(f"Error capturing frame {i}")
            else:
                print(f"Error arduino stepper")

        print("Waiting for threads to finish")
        processing_queue.join()

        print("Waiting for ML threads to finish")
        ml_queue.join()

        all_3d_points = []

        global_poses = {}

        global_poses[0] = {
            'R': np.eye(3),
            't': np.zeros((3, 1))
        }

        for i in range(config.TOTAL_PHOTOS - 1):
            pair_name = f"{i}_{i + 1}"

            match_data = all_matches.get(pair_name)

            if match_data is None:
                print(f"[CHAINING] pair_name is missing")
                continue 

            R_local = match_data['R_local']
            t_local = match_data['t_local']
            pts1 = match_data['ptsA']
            pts2 = match_data['ptsB']

            R_prev = global_poses[i]['R']
            t_prev = global_poses[i]['t']

            R_global = R_local @ R_prev
            t_global = R_local @ t_prev + t_local

            global_poses[i + 1] = {
                'R': R_global,
                't': t_global
            }

            P1 = np.hstack((R_prev, t_prev))
            P2 = np.hstack((R_global, t_global))

            K_inv = np.linalg.inv(K)

            for j in range(len(pts1)):
                pt1_homogeneus = np.array([pts1[j, 0], pts1[j, 1], 1.0])
                pt2_homogeneus = np.array([pts2[j, 0], pts2[j, 1], 1.0])

                pt1_normalized = K_inv @ pt1_homogeneus
                pt2_normalized = K_inv @ pt2_homogeneus

                pt_3d = triangulate_points(pt1_normalized, pt2_normalized, P1, P2)

                all_3d_points.append(pt_3d)

        print(f"[MAIN] Reconstruction done with {len(all_3d_points)} points")   

    finally:
        arduino.close()
        camera.close()

        for _ in range(config.THREADS_NR):
            processing_queue.put(None)

        for t in threads:
            t.join()

        for _ in range(config.ML_THREADS_NR):
            ml_queue.put(None)

        for t in ml_threads:
            t.join()

        save_point_cloud_ply(all_3d_points)


if __name__ == "__main__":
    main()