"""
@author: Michael
@version: 2021-04-24
"""
from __future__ import print_function

import time

from image_use_pyopenpose import *
from Modules.common import draw_str, clock, StatValue
import cv2

'''
Multithreaded video processing sample.
Usage:
   video_threaded.py {<video device number>|<video file name>}
   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.
Keyboard shortcuts:
   ESC - exit
   space - switch between multi and single threaded processing
'''

# Python 2/3 compatibility

import numpy as np
import cv2 as cv

from multiprocessing.pool import ThreadPool
from collections import deque


def main():
    cap = cv2.VideoCapture(1)
    model_dir = "../models"
    json_dir = "../tmp_json"
    net_resolution = "-1x192"
    threadn = cv.getNumberOfCPUs()

    op = PyOpenpose(net_resolution, model_dir)
    op.openpose_start()

    i = 0

    def process_frame(frame, t0, i):
        # some intensive computation...
        # frame = cv.medianBlur(frame, 19)
        # frame = cv.medianBlur(frame, 19)
        datum = op.openpose_detection(frame, None, None)
        # cv.imshow('opencv', datum.cvOutputData)`
        frame = datum.cvOutputData
        return frame, t0

    pool = ThreadPool(processes=threadn)
    pending = deque()

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        print(len(pending))
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
            cv.imshow('threaded video', res)
        print(len(pending))
        if len(pending) < threadn:
            _ret, frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            task = pool.apply_async(process_frame, (frame.copy(), t, i))
            i = (i + 1) % threadn
            pending.append(task)
        ch = cv.waitKey(1)
        if ch == 27:
            break
    cap.release()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
