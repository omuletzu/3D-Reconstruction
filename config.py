import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARDUINO_PORT = 'COM3'
CAMERA_INDEX = 'http://10.247.131.204:8080/video'
TOTAL_PHOTOS = 16

DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'temple_ring_sparse_dataset')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'data', 'camera_params_temple.json')
MODEL_PATH =  os.path.join(BASE_DIR, 'ml_matching', 'feature-descriptor.keras')
SAVE_POINT_CLOUD_PATH = os.path.join(BASE_DIR, 'point_cloud.ply')

DATA_FOLDER_IMAGES_EXTENSION = 'png'

PREPROCESSING_THREADS_NR = 8
ML_THREADS_NR = 1
GEOMETRY_THREADS_NR = 4

RANSAC_MAX_ITERS = 1000
RANSAC_THRESHOLD = 0.01

MAX_MATCHES = 7500

PATCH_SIZE = 32
BATCH_SIZE = 32

MAX_WIDTH_CALIBRATION = 640

USE_CACHE = True
CACHE_PATH = "sfm_data_cache.pkl"