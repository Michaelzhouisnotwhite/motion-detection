"""
@author: Michael
@version: 2021-06-27
"""
import cv2 as cv
import numpy as np
import Openpose.pyopenpose as op
from Modules.common import draw_str, StatValue, clock
from Modules.read_cam import ReadCam


class PyOpenPose(object):
    def __init__(self, net_resolution="-1x192", model_folder="../models", display=1, render_pose=1):
        self.params = dict()
        self.opWrapper = None
        self.net_resolution = net_resolution
        self.model_folder = model_folder
        self.datum = None
        self.json_dir = ""
        self.file_name = ""
        self.key_points_list = []

        self.display = display
        self.render_pose = render_pose

        self.output = None

    def start(self):
        self.opWrapper = op.WrapperPython()
        self.__openpose_params_settings()
        self.datum = op.Datum()
        self.opWrapper.start()

    def __openpose_params_settings(self):
        self.params["net_resolution"] = self.net_resolution
        # max:208 4.4fps
        self.params["model_pose"] = "BODY_25"
        self.params["model_folder"] = self.model_folder
        self.params["display"] = self.display
        self.params["render_pose"] = self.render_pose
        self.params["face"] = False
        self.params["hand"] = False
        # self.params["disable_blending"] = True

        self.opWrapper.configure(self.params)

    def detection(self, image) -> op.Datum:
        imageToProcess = image
        self.datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        self.output = self.datum.cvOutputData

        return self.datum

    def key_points_dict(self) -> list:
        key_empty = np.zeros((25, 3))
        key_points_list = []
        if self.datum.poseKeypoints is None:
            content = {"person id": None, "body_key_points": key_empty}
            key_points_list.append(content)
        else:
            for i, key_points in enumerate(self.datum.poseKeypoints):
                key_points = np.around(key_points.copy(), decimals=2)
                np.set_printoptions(precision=3)
                np.set_printoptions(suppress=True)
                content = {"person id": i, "body_key_points": key_points}
                key_points_list.append(content)
        self.key_points_list = key_points_list
        return key_points_list


if __name__ == "__main__":
    pyOpenPose = PyOpenPose()
    pyOpenPose.start()

    cam = ReadCam(1)

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()

    while True:
        frame = cam.readFrame()
        t = clock()
        frame_interval.update(t - last_frame_time)
        last_frame_time = t
        pyOpenPose.detection(frame)
        print(pyOpenPose.key_points_dict())

        output = pyOpenPose.output
        latency.update(clock() - t)
        draw_str(output, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
        draw_str(output, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))

        draw_str(frame, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
        draw_str(frame, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))

        cv.imshow("demo-raw", frame)
        cv.imshow("demo-cooked", output)
        if cv.waitKey(1) == 27:
            break
    cam.exit()
    cv.destroyAllWindows()
