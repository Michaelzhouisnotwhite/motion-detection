import sys
import cv2 as cv
import cv2
import os
import numpy as np
import json
import time
from sys import platform
import argparse
import concurrent.futures
# sys.path.extend(['E:\\User_Michaels\\PYTHON\\Openpose\\openpose-learning_demo\\cv-useCoCo-GPU', 'E:/User_Michaels'
#                                                                                                 '/PYTHON/Openpose'
#                                                                                                 '/openpose-learning_demo/cv-useCoCo-GPU'])
# Import Openpose (Windows/Ubuntu/OSX)
# try:
#     # Change these variables to point to the correct folder (Release/x64 etc.)
#     sys.path.append(dir_path + '/bin/python/openpose/Release')
#     os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' + dir_path + '/bin;'
#     import pyopenpose as op
# except ImportError as e:
#     print(
#         'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
#         'script in the right folder?')
#     raise e
import openpose as op


def to_precised_decimal(mat, num):
    np.set_printoptions(suppress=True)
    return np.round(mat, decimals=num)


class PyOpenpose(object):
    def __init__(self, net_resolution, model_folder = op.ModelDir):
        """
        :return:
        """
        self.params = dict()
        self.opWrapper = op.WrapperPython()
        self.net_resolution = net_resolution
        self.model_folder = model_folder

        self.datum = op.Datum()
        self.opWrapper = op.WrapperPython()
        self.openpose_params_settings()
        self.json_dir = ""
        self.file_name = ""
        self.jFile = []

    def openpose_params_settings(self):

        self.params["net_resolution"] = self.net_resolution
        # max:208 4.4fps
        self.params["model_pose"] = "BODY_25"
        self.params["model_folder"] = self.model_folder
        # params["display"] = 0
        # params["render_pose"] = 0
        self.params["face"] = False
        self.params["hand"] = False
        # self.params["disable_blending"] = True

        self.opWrapper.configure(self.params)

    def openpose_start(self):
        self.opWrapper.start()

    def openpose_detection(self, frame, file_dir, write_json):

        # resize

        imageToProcess = frame
        self.datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        # key_points = np.round(datum.poseKeypoints, decimals=2)
        # np.set_printoptions(suppress=True)

        # print("Body keypoints:")
        # print(key_points)
        if write_json is not None:
            self.write_key_points_json(file_dir, write_json)
        return self.datum

    def write_key_points_json(self, *file_name):

        with open(file_name[0] + "\\" + file_name[1] + ".json", "w") as f:
            json.dump(self.jFile, f)

    def get_key_points_dict(self):
        jFile = list()
        if self.datum.poseKeypoints is None:
            content = {"person id": None, "body_key_points": []}
            jFile.append(content)
        else:
            for i, key_points in enumerate(self.datum.poseKeypoints):
                key_points = np.round(key_points.copy(), decimals=2).tolist()
                content = {'person id': i, 'body_key_points': key_points}
                jFile.append(content)
        self.jFile = jFile
        return jFile

    def getKeyPoints(self):
        key_empty = np.zeros((25, 3))
        if self.datum.poseKeypoints is None:
            return key_empty
        return self.datum.poseKeypoints


if __name__ == "__main__":
    # model_dir = "../models"
    json_dir = "../tmp_json"
    cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FPS, 2)

    while 1:
        # time.sleep(0.7)
        ret, image = cap.read()
        if not ret:
            print("no cam")
            break
        # image = cv.resize(image.copy(), (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
        # datum = openpose_detection(image, "-1x192", model_dir, json_dir)
        # cv.imshow('opencv', datum.cvOutputData)
        if cv.waitKey(1) == ord("q") or cv.waitKey(1) == 27:
            break

# --------------------------------------------------------------
# target_path = "Datasets/obj_images/fall"
# for i, file in enumerate(os.listdir("Datasets/obj_images/fall")):
#     if file.endswith('.jpg'):
#
#         cv2.imshow('test', datum.cvOutputData)
#         cv2.waitKey(0)
# --------------------------------------------------------------
