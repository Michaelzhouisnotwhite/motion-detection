"""
@author: Michael
@version: 2021-06-26
"""
import copy
import threading

import cv2 as cv

from multiprocessing.pool import ThreadPool
from multiprocessing import Process
from collections import deque
from Modules.PyOpenPose import PyOpenPose
from Modules.read_cam import ReadCam
from threading import Thread


class Signal:
    empty = 1
    mutex = threading.Lock()
    pending = deque()


pool = ThreadPool()


class CamProcess(ReadCam):
    def __init__(self, device_no=0):
        ReadCam.__init__(self, device_no)

    def run(self):
        while True:
            Signal.mutex.acquire()
            print(len(Signal.pending))
            if len(Signal.pending) < 1:
                print(len(Signal.pending))
                frame = self.readFrame()
                Signal.pending.append(copy.deepcopy(frame))

                Signal.mutex.release()

                print(self.readFrame().shape)
                cv.imshow("cam-process", frame)
                cv.waitKey(1)

    def exit(self):
        super().exit()


if __name__ == "__main__":
    pass