#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib
import pprint

import pandas as pd

# Get from arg_parsing_utils import check_for_missing_args, parse_args
from image_analysis_2D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[ ]:


patients_dir = pathlib.Path(f"{image_base_dir}/data/").resolve(
    strict=True
)  # directory containing patient folders

patient_list_file = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(patient_list_file, header=None)[0].tolist()

# get a list of patient directories
patient_dirs = [d for d in patients_dir.iterdir() if d.is_dir() and d.name in patients]

patient_dirs.sort()


# In[ ]:


# get a list of the well_fov directories for each patient
well_fov_dirs = [
    x
    for patient_dir in patient_dirs
    for x in pathlib.Path(f"{patient_dir}/2D_analysis/0b.middle_slice").iterdir()
    if x.is_dir()
]
well_fov_dirs.sort()
well_fov_df = pd.DataFrame(well_fov_dirs, columns=["dir_path"])
well_fov_df["patient"] = well_fov_df["dir_path"].apply(
    lambda x: str(x.parent).split("/")[-3]
)
well_fov_df["well_fov"] = well_fov_df["dir_path"].apply(lambda x: x.stem)
well_fov_df


# In[ ]:


present_files = 0
missing_files = 0
missing_files_list = []
for index, row in well_fov_df.iterrows():
    patient = row["patient"]
    well_fov = row["well_fov"]
    max_z_slice_dir_to_check = pathlib.Path(
        f"{image_base_dir}/data/{patient}/2D_analysis/2a.cellprofiler_zmax_proj_output/{well_fov}/"
    ).resolve()
    middle_slice_dir_to_check = pathlib.Path(
        f"{image_base_dir}/data/{patient}/2D_analysis/2b.cellprofiler_middle_slice_output/{well_fov}/"
    ).resolve()
    middle_n_slice_dir_to_check = pathlib.Path(
        f"{image_base_dir}/data/{patient}/2D_analysis/2c.cellprofiler_middle_n_slice_max_proj_output/{well_fov}/"
    ).resolve()
    if not max_z_slice_dir_to_check.is_dir():
        missing_files += 1
        missing_files_list.append(str(max_z_slice_dir_to_check))
    else:
        present_files += 1
    if not middle_slice_dir_to_check.is_dir():
        missing_files += 1
        missing_files_list.append(str(middle_slice_dir_to_check))
    else:
        present_files += 1
    if not middle_n_slice_dir_to_check.is_dir():
        missing_files += 1
        missing_files_list.append(str(middle_n_slice_dir_to_check))
    else:
        present_files += 1


# In[ ]:


reruns_df = pd.DataFrame(missing_files_list, columns=["dir_path"])
reruns_df["patient"] = reruns_df["dir_path"].apply(lambda x: str(x).split("/")[-4])
reruns_df["well_fov"] = reruns_df["dir_path"].apply(lambda x: str(x).split("/")[-1])
reruns_df.drop(columns=["dir_path"], inplace=True)
reruns_df.drop_duplicates(inplace=True)
pathlib.Path("../loadfiles/").mkdir(parents=True, exist_ok=True)
reruns_df.to_csv(
    f"../loadfiles/featurization_loadfile.txt", index=False, sep="\t", header=False
)


# In[ ]:


print(f"Total directories checked: {len(well_fov_df) * 3}")
print(f"Present directories: {present_files}")
print(f"Missing directories: {missing_files}")
print("Missing directories list:")
if missing_files < 50:
    pprint.pprint(missing_files_list)
