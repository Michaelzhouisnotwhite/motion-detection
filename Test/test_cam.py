"""
@author: Michael
@version: 2021-06-26
"""
# !/usr/bin/env python
from __future__ import print_function

import time

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

from Modules.common import clock, draw_str, StatValue
import video


class DummyTask:
    def __init__(self, data):
        self.data = data

    def ready(self):
        return True

    def get(self):
        return self.data


def main():
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    # cap = video.create_capture(fn)
    """
    @Michael
    """
    cap = video.create_capture(1)

    def process_frame(frame, t0):
        # some intensive computation...
        # frame = cv.medianBlur(frame, 19)
        # frame = cv.medianBlur(frame, 19)
        # frame = cv.medianBlur(frame, 19)
        # frame = cv.medianBlur(frame, 19)
        time.sleep(1)
        return frame, t0

    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    threaded_mode = True

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        print("before output: ", len(pending))
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
            cv.imshow('threaded video', res)
        print("after output: ", len(pending))
        if len(pending) < threadn:
            _ret, frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, t))
            pending.append(task)
        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
