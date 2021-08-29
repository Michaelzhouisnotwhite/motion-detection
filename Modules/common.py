"""
@author: Michael
@version: 2021-06-26
"""
import cv2
import cv2 as cv
from collections import deque


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


def __main():
    return clock()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


class StatValue:
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0 - c) * v


class Config:
    cpu_nums = cv.getNumberOfCPUs()


if __name__ == "__main__":
    __main()
    # while True:
    # print(__main())
