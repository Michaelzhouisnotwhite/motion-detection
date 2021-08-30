"""
@author: Michael
@version: 2021-04-25
"""

import sys
from multiprocessing.pool import ThreadPool
from utils.common import draw_str, clock, StatValue
from collections import deque
from image_use_pyopenpose import *
import cv2 as cv
import os

import numpy as np


class config:
    def __init__(self, file_name):
        self.root = r"..\Datasets/fenix falling data"
        self.image_dir_path = os.path.join(self.root, file_name)
        self.rate = float(60)
        self.image_path_list = [os.path.join(self.image_dir_path, image_path)
                                for image_path in os.listdir(self.image_dir_path)]

    def getIntervals(self) -> float:
        return 1 / self.rate

    def getT_2(self):
        return self.getIntervals() ** 2


def get_acc(dis, t_2):
    dis[:, 0] = dis[:, 0] / t_2
    return dis


class Process(PyOpenpose):
    def __init__(self):
        model_dir = op.ModelDir
        json_dir = "../tmp_json"
        net_resolution = "-1x192"
        super().__init__(net_resolution, model_dir)
        self.openpose_start()

    def run(self, img, t):
        datum = self.openpose_detection(img, None, None)
        img = datum.cvOutputData
        pose = datum.poseKeypoints
        return pose, t, img


# for only one person
def pose_distance(key_points_0, key_points_1) -> np.ndarray or int:
    n = np.zeros((25, 2))
    n = n * np.nan
    if key_points_0 is None or key_points_1 is None:
        return n
    key_points_0[:, :, :2][key_points_0[:, :, 2] == 0] = np.NAN
    X = np.add(key_points_0[0, :, 0], -key_points_1[0, :, 0])
    Y = np.add(key_points_0[0, :, 1], -key_points_1[0, :, 1])
    Confidence = np.add(key_points_0[0, :, 2], key_points_1[0, :, 2]) / 2
    dis = np.sqrt(np.add(np.square(X), np.square(Y)))
    dis_arr = np.hstack((dis.reshape(25, 1), Confidence.reshape(25, 1)))
    return dis_arr


def main():
    data_dir = "fall-16-cam0-rgb"
    conf = config(data_dir)
    t_2 = conf.getT_2()

    dq = deque()

    cpus = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=cpus)

    p = Process()

    images_list = []
    for image in conf.image_path_list:

        # while True:
        while len(dq) > 1 and dq[0].ready():
            pose0, t0, d0 = dq.popleft().get()
            pose1, t1, d1 = dq[0].get()
            through = (t0 - t1)
            dis = pose_distance(pose0, pose1)
            a = get_acc(dis, t_2)
            # else:
            frame = d0
            # draw_str(frame, (20, 40), "a: %f" % a[0, 0])
            cv.imshow("video", frame)
            print(to_precised_decimal(a, 4))

        if len(dq) < cpus:
            frame = cv.imread(image)
            t = clock()
            task = pool.apply_async(p.run, (frame.copy(), t))
            dq.append(task)
        ch = cv.waitKey(1)
        if ch == 27:
            break


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
