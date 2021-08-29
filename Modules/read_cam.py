"""
@author: Michael
@version: 2021-06-26
"""
import cv2 as cv
import numpy as np
from threading import Thread


class ReadCam(object):
    def __init__(self, device_no=0):
        self.cap = cv.VideoCapture(device_no)

    def readFrame(self) -> np.ndarray:
        if self.cap.isOpened() is False:
            exit(cv.Error)
        elif self.cap.isOpened() is True:
            __ret, frame = self.cap.read()
            return frame

    def __del__(self):
        self.exit()

    def exit(self):
        self.cap.release()


if __name__ == "__main__":
    class CamProcess(ReadCam):
        def __init__(self, device_no=0):
            ReadCam.__init__(self, device_no)

        def run(self):
            while True:
                frame = self.readFrame()
                print(self.readFrame().shape)
                cv.imshow("cam-process", frame)
                cv.waitKey(1)

        def exit(self):
            super().exit()
    # class ShowProcess(Pro)

    cam_process = CamProcess(1)
    Thread(target=cam_process.run).start()

    r = ReadCam(1)
    while True:
        cv.imshow("r", cam_process.readFrame())
        cv.waitKey(1)
