import cv2
import time

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.cap.isOpened():
            print("Cannot access USB camera")

    def captura_frame(self):
        for _ in range(5):
            ret, frame = self.cap.read()

        if not ret:
            print("Error capturing frame")
            return None
        
        return frame
    
    def close(self):
        self.cap.release()