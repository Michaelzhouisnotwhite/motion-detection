"""
@author: Michael
@version: 2021-09-06
"""
# %%
import cv2 as cv
import os

workspace_root = r"E:\User_Michaels\My Projects\Python Project\motion-detection"
# %%
chute01_video_folder_path = os.path.join(workspace_root, "Datasets/dataset/chute01")
chute01_pic_folder_path = os.path.join(workspace_root, "chu_data/video2pic")
# %%
chute01_video_list = []
for avi in os.listdir(chute01_video_folder_path):
    chute01_video_list.append(os.path.join(chute01_video_folder_path, avi))

# %%

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
        (_, video_folder) = os.path.split(filepath)
        (file_name, extension) = os.path.splitext(temp_filepath)
        count += 1
        if count % 10 == 0:
            save_path = os.path.join(chute01_pic_folder_path, video_folder) + "/{}/frame{}.png".format(file_name, no)
            no += 1
            cv.imwrite(save_path, frame)
        continue

    cap.release()
cv.destroyAllWindows()
