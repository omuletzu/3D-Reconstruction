import os
import time
import threading
import queue
import cv2
from hardware.camera import Camera
from hardware.stepper import ArduinoController
from vision2d.processing import fix_distorsion, remove_background, enhance_contrast
from ml_matching.inference import extract_patches, compute_descriptors_for_patches, match_images

TOTAL_PHOTOS = 18

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

def ai_and_matching_worker(model):
    while True:
        idx = ml_queue.get()

        if idx is None:
            break

        print(f"[ML THREAD] infering for image ${idx}")

        patches = processed_data[idx]['patches']

        descriptors = compute_descriptors_for_patches(patches, model)

        with lock:
            processed_data[idx]['descriptors'] = descriptors

        neighbours_to_check = [
            (idx - 2) % TOTAL_PHOTOS,
            (idx - 1) % TOTAL_PHOTOS,
            (idx + 1) % TOTAL_PHOTOS,
            (idx + 2) % TOTAL_PHOTOS
        ]

        for neighbour in neighbours_to_check:
            if neighbour == (idx + 1) % TOTAL_PHOTOS or neighbour == (idx + 2) % TOTAL_PHOTOS:
                img_A, img_B = idx, neighbour
            else:
                img_A, img_B = neighbour, idx

            pair_name = f"${img_A}_${img_B}"

            with lock:
                if pair_name in all_matches:
                    continue

                ready_A = (img_A in processed_data) and (processed_data[img_A].get('descriptors') is not None)

                ready_B = (img_B in processed_data) and (processed_data[img_B].get('descriptors') is not None)

                ready = ready_A and ready_B

            if ready:
                print(f"[MATCHING] Starting for ${pair_name}")

                descA = processed_data[img_A]['descriptors']
                kpA = processed_data[img_A]['keypoints']

                descB = processed_data[img_B]['descriptors']
                kpB = processed_data[img_B]['keypoints']

                ptsA, ptsB = match_images(descA, kpA, descB, kpB)

                with lock:
                    all_matches[pair_name] = (ptsA, ptsB)

                print(f"[MATCHING] Done for ${pair_name}")

        ml_queue.task_done()

def main():
    ARDUINO_PORT = 'COM3'
    CAMERA_INDEX = 1
    DATA_FOLDER = 'data/raw_captures'
    CALIBRATION_FILE = 'hardware/camera_params.json'

    THREADS_NR = 8

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    print("Hardware initialization")

    arduino = ArduinoController(port=ARDUINO_PORT)
    camera = Camera(camera_index=CAMERA_INDEX)

    if not os.path.exists(CALIBRATION_FILE):
        camera.calibrate_camera()

    threads = []

    for _ in range(THREADS_NR):
        t = threading.Thread(target=thread_worker)

        t.start()

        threads.append(t)

    time.sleep(2)

    print("Started photo scanning")

    try:
        for i in range(TOTAL_PHOTOS):
            print(f"Capturing frame ${i}")

            if arduino.rotate_step():
                time.sleep(0.5)

                image = camera.captura_frame()

                if image:
                    image_file_path = os.path.join(DATA_FOLDER, f"photo_${i}.jpg")

                    cv2.imwrite(image_file_path, image)

                    processing_queue.put((image_file_path, i))

                    print(f"Saved image ${i}")
                else:
                    print(f"Error capturing frame ${i}")
            else:
                print(f"Error arduino stepper")

        print("Waiting for threads to finish")
        processing_queue.join()

    finally:
        arduino.close()
        camera.close()

        for _ in range(THREADS_NR):
            processing_queue.put(None)

        for t in threads:
            t.join()

if __name__ == "__main__":
    main()