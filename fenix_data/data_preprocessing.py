# %%
from operator import index
import numpy as np
import os
import pandas as pd

# %%

pic_dir = r"E:\User_Michaels\My Projects\Python Project\motion-detection\Datasets\fenix falling data"
fall_dir_list = [path for path in os.listdir(pic_dir)]

# %%
csv_dir = r"E:\User_Michaels\My Projects\Python Project\motion-detection\fenix_data\csv"
# %%
csv_name_list = fall_dir_list
# %%
full_pic_name_list = []
for folder in fall_dir_list:
    folder_path = os.path.join(pic_dir, folder)
    pic_name_list = []
    for pic in os.listdir(folder_path):
        (filepath, tempfilename) = os.path.split(pic)
        (shortname, extension) = os.path.splitext(tempfilename)
        pic_name_list.append([shortname, extension])
    full_pic_name_list.append(pic_name_list)
# %%
df_save_list = []
for csv_name, pic_n_list in zip(csv_name_list, full_pic_name_list):
    arr = np.array(pic_n_list)
    f = np.zeros_like(arr[:, 0])[:, None]
    df = pd.DataFrame(columns=["data name", "extension", "fall"], data=np.concatenate((arr, f), axis=1))
    df_save_list.append(df)
# %%
for csv_name, df in zip(csv_name_list, df_save_list):
    # f =pd.ExcelWriter(os.path.join(csv_dir, csv_name+".xlsx"))
    path = os.path.join(csv_dir, csv_name + ".csv")

    df.to_csv(path_or_buf=path, index=False)

# %%
