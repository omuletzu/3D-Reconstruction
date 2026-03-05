import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARDUINO_PORT = 'COM3'
CAMERA_INDEX = 1
TOTAL_PHOTOS = 18

DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'raw_captures')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'hardware', 'camera_params.json')
MODEL_PATH =  os.path.join(BASE_DIR, 'ml_matching', 'feature-descriptor.keras')
SAVE_POINT_CLOUD_PATH = os.path.join(BASE_DIR, 'point_cloud.ply')

THREADS_NR = 8
ML_THREADS_NR = 1

RANSAC_MAX_ITERS = 1000
RANSAC_THRESHOLD = 0.01

MAX_MATCHES = 1000

PATCH_SIZE = 32
BATCH_SIZE = 32