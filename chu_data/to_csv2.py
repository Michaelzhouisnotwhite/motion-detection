"""
@author: Michael
@version: 2021-09-06
"""
# %%
import pandas as pd
import numpy as np
import os

# %%
# 1. workspace_root = 右键-> copy-> absolutely path
workspace_root = r"E:\User_Michaels\My Projects\Python Project\motion-detection"
# %%

# 2. 改数字
pic_dir = os.path.join(workspace_root, "chu_data/video2pic/chute03")
csv_saved_root_path = os.path.join(workspace_root, "chu_data/csv")


# %%
def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


# %%
csv_saved_chute01 = os.path.join(csv_saved_root_path, os.path.split(pic_dir)[1])
mkdir(os.path.join(csv_saved_chute01))
# %%
all_pic_list = []
for cam_dir in os.listdir(pic_dir):
    save_csv_file_name = os.path.join(csv_saved_chute01, cam_dir) + ".csv"
    cam_dir_abspath = os.path.join(pic_dir, cam_dir)
    pic_list = []
    for pic_name in os.listdir(cam_dir_abspath):
        name, extension = os.path.splitext(pic_name)
        pic_list.append([name, extension, np.nan])
    all_pic_list.append(pic_list)
    df = pd.DataFrame(data=np.array(pic_list), columns=["pic name", "ext", "fall"])
    df.to_csv(save_csv_file_name, index=False)

# %%
