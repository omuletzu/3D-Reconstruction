import serial
import time

class ArduinoController:
    def __init__(self, port='COM3', baudrate=9600):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=2)

            time.sleep(2)

            print(f"Connected to port ${port}")

        except Exception as e:
            print(f"Error connecting ${e}")

    def rotate_step(self):
        self.ser.write(b'R')

        while True:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode('utf-8').strip()

                if response == 'D':
                    print("Rotated stepper")
                    return True
                
            time.sleep(0.1)

    def close(self):
        self.ser.close()