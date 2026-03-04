import os
import time
import threading
import queue
import cv2
from hardware.camera import Camera
from hardware.stepper import ArduinoController
from vision2d.processing import fix_distorsion, remove_background, enhance_contrast

from tensorflow.keras.models import load_model

processing_queue = queue.Queue()

processed_data = {}

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

        fast = cv2.FastFeatureDetector_create(threshold=30)

        keypoints = fast.detect(image, None)

        with lock:
            processed_data[idx] = {
                'image': image,
                'keypoints': keypoints,
            }

        print(f"[THREAD] Processed image {idx}")
        
        processing_queue.task_done()


def main():
    ARDUINO_PORT = 'COM3'
    CAMERA_INDEX = 1
    TOTAL_PHOTOS = 18
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