"""
@author: Michael
@version: 2021-09-17
"""

# %%
import cv2 as cv
import os
# 1. workspace_root = 右键-> copy-> absolutely path
workspace_root = r"E:\User_Michaels\My Projects\Python Project\motion-detection"
# %%

for i in range(1, 25):
    if i < 10:
        chute01_video_folder_path = os.path.join(workspace_root, "Datasets/chu_data/chute0{}".format(i))
    else:
        chute01_video_folder_path = os.path.join(workspace_root, "Datasets/chu_data/chute{}".format(i))

    chute01_pic_folder_path = os.path.join(workspace_root, "chu_data/video2pic")
    # %%
    chute01_video_list = []
    for avi in os.listdir(chute01_video_folder_path):
        chute01_video_list.append(os.path.join(chute01_video_folder_path, avi))

    # %%
    for path in chute01_video_list:
        cap = cv.VideoCapture(path)
        count = 0
        no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            (filepath, temp_filepath) = os.path.split(path)
            (video_kind_folder, video_folder) = os.path.split(filepath)
            (file_name, extension) = os.path.splitext(temp_filepath)
            count += 1

            _, kind_name = os.path.split(video_kind_folder)
            if count % 10 == 0:
                save_path = os.path.join(chute01_pic_folder_path, video_folder)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                if not os.path.exists(save_path + "/{}".format(file_name)):
                    os.mkdir(save_path + "/{}".format(file_name))
                save_pic_path = save_path + "/{}/frame{}.png".format(file_name, no)
                no += 1
                cv.imwrite(save_pic_path, frame)
                print("chute {}/{}/frame{}".format(i, file_name, no), "complete")
            continue

        cap.release()
    cv.destroyAllWindows()
