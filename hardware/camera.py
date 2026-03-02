import cv2
import numpy as np
import json
import glob

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.cap.isOpened():
            print("Cannot access USB camera")

    def calibrate_camera(self):
        BOARD_SIZE = (7, 6)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)

        objp[:, :2] = np.mgrid[0 : BOARD_SIZE[0], 0 : BOARD_SIZE[1]].T.reshape(-1, 2)

        objpoints = [] 
        imgpoints = []

        images = glob.glob('/calibration_data/*.jpg')

        for fname in images:
            img = cv2.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (BOARD_SIZE[0], BOARD_SIZE[1]), None)

            if ret == True:
                objpoints.append(objp)

                corners_detailed = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                
                imgpoints.append(corners_detailed)

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            calib_results = {
                'camera_matrix': mtx.toList(),
                'distortion_params': dist.toList()
            }

            with open('camera_params.json', 'w') as f:
                json.dump(calib_results, f)

            print("Succesfully found intrinsec camera matrix and distorsion parameters")


    def captura_frame(self):
        for _ in range(5):
            ret, frame = self.cap.read()

        if not ret:
            print("Error capturing frame")
            return None
        
        return frame
    
    def close(self):
        self.cap.release()