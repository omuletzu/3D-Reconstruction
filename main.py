import config

import os
import time
import threading
import queue
import json
import numpy as np
import pickle
import cv2
from hardware.camera import Camera
from hardware.stepper import ArduinoController
from ml_matching.inference import load_model
from workers.preprocessing_worker import preprocessing_worker
from workers.matching_worker import matching_worker
from workers.geometry_worker import geometry_worker
from geometry3d.global_reconstruction import global_reconstruction

preprocessing_queue = queue.Queue()
matching_queue = queue.Queue()
geometry_queue = queue.Queue()

processed_data = {}
all_matches = {}
lock = threading.Lock()

def serialize_keypoints(kp_list):
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_list]

def deserialize_keypoints(kp_data):
    return [cv2.KeyPoint(x=d[0], y=d[1], size=d[2], angle=d[3], response=d[4], octave=d[5], class_id=d[6]) for d in kp_data]

def main():
    global processed_data, all_matches

    if not os.path.exists(config.DATA_FOLDER):
        os.makedirs(config.DATA_FOLDER)

    print("Hardware initialization")

    # arduino = ArduinoController(port=config.ARDUINO_PORT)
    # camera = Camera(camera_index=config.CAMERA_INDEX)

    # if not os.path.exists(config.CALIBRATION_FILE):
    #     camera.calibrate_camera()

    with open(config.CALIBRATION_FILE, 'r') as f:
        cam_data = json.load(f)

        K = np.array(cam_data['camera_matrix'])

    model = load_model(config.MODEL_PATH)

    preprocessing_threads = []
    matching_threads = []
    geometry_threads = []

    if config.USE_CACHE and os.path.exists(config.CACHE_PATH):
        with open(config.CACHE_PATH, "rb") as f:
            cached_data = pickle.load(f)
            
        all_matches.update(cached_data["all_matches"])
        
        loaded_p_data = cached_data["processed_data"]

        for idx in loaded_p_data:
            if 'keypoints' in loaded_p_data[idx] and loaded_p_data[idx]['keypoints'] is not None:
                loaded_p_data[idx]['keypoints'] = deserialize_keypoints(loaded_p_data[idx]['keypoints'])
        
        processed_data.update(loaded_p_data)
        K = cached_data["K"]

    else:
        model = load_model(config.MODEL_PATH)

        for _ in range(config.PREPROCESSING_THREADS_NR):
            t = threading.Thread(target=preprocessing_worker,
                                args=(preprocessing_queue, lock, processed_data, matching_queue))
            t.start()
            preprocessing_threads.append(t)

        for _ in range(config.ML_THREADS_NR):
            t = threading.Thread(target=matching_worker, 
                                args=(model, matching_queue, processed_data, lock, all_matches, geometry_queue))
            t.start()
            matching_threads.append(t)

        for _ in range(config.GEOMETRY_THREADS_NR):
            t = threading.Thread(target=geometry_worker, 
                                args=(K, geometry_queue, lock, all_matches))
            t.start()
            geometry_threads.append(t)

    time.sleep(2)

    print("Started photo scanning")

    # try:
    #     for i in range(config.TOTAL_PHOTOS):
    #         print(f"Capturing frame {i}")

    #         if arduino.rotate_step():
    #             time.sleep(0.5)

    #             image = camera.captura_frame()

    #             if image:
    #                 image_file_path = os.path.join(config.DATA_FOLDER, f"photo_{i}.${config.DATA_FOLDER_IMAGES_EXTENSION}")

    #                 cv2.imwrite(image_file_path, image)

    #                 preprocessing_queue.put((image_file_path, i))

    #                 print(f"Saved image {i}")
    #             else:
    #                 print(f"Error capturing frame {i}")
    #         else:
    #             print(f"Error arduino stepper")

    try:
        if not (config.USE_CACHE and os.path.exists(config.CACHE_PATH)):
            print("Started photo scanning")
            for i in range(config.TOTAL_PHOTOS):
                image_file_path = os.path.join(config.DATA_FOLDER, f"templeSR{(i + 1):04d}.{config.DATA_FOLDER_IMAGES_EXTENSION}")
                preprocessing_queue.put((image_file_path, i))

            print("Waiting for threads to finish...")
            preprocessing_queue.join()
            matching_queue.join()
            geometry_queue.join()

            save_ready_processed_data = {}
            for idx, data in processed_data.items():
                save_ready_processed_data[idx] = data.copy()
                if 'keypoints' in data and data['keypoints'] is not None:
                    save_ready_processed_data[idx]['keypoints'] = serialize_keypoints(data['keypoints'])

            data_to_save = {
                "all_matches": dict(all_matches),
                "processed_data": save_ready_processed_data,
                "K": K
            }

            with open(config.CACHE_PATH, "wb") as f:
                pickle.dump(data_to_save, f)

        global_reconstruction(all_matches, processed_data, K)

    finally:
        for q, threads in [(preprocessing_queue, preprocessing_threads), 
                          (matching_queue, matching_threads), 
                          (geometry_queue, geometry_threads)]:
            for _ in threads: q.put(None)
            for t in threads: t.join()


if __name__ == "__main__":
    main()