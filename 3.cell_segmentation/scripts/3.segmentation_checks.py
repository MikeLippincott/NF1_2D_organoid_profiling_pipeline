#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import pandas as pd
import tifffile
import tqdm
from image_analysis_2D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_2D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


patient_ids_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patient_ids = patient_ids_path.read_text().splitlines()
patient_ids


# In[3]:


illumcorr_paths = [
    "0a.zmax_proj",
    "0b.middle_slice",
    "0c.middle_n_slice_max_proj",
]
masks = ["organoid_mask", "nuclei_mask", "cell_mask"]


# In[4]:


expected_count = 0
missing_count = 0
missing_dict = {
    "patient_id": [],
    "twoD_method": [],
    "well_fov": [],
    "mask": [],
    "missing_mask_path": [],
}

for patient in tqdm.tqdm(patient_ids, desc="Patients"):
    for illum_corr_sub_path in illumcorr_paths:
        illum_corr_path = pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/{illum_corr_sub_path}/"
        )
        well_fovs = [wf.stem for wf in illum_corr_path.glob("*")]
        for well_fov in well_fovs:
            if well_fov == "run_stats":
                continue
            for mask in masks:
                mask_path = pathlib.Path(
                    f"{image_base_dir}/data/{patient}/2D_analysis/{illum_corr_sub_path}/{well_fov}/{well_fov}_{mask}.tiff"
                ).resolve()
                if "0a" in str(illum_corr_path.name):
                    twoD_method = "zmax"
                elif "0b" in str(illum_corr_path.name):
                    twoD_method = "middle"
                elif "0c" in str(illum_corr_path.name):
                    twoD_method = "middle_n"
                expected_count += 1
                if not mask_path.exists():
                    missing_count += 1
                    missing_dict["patient_id"].append(patient)
                    missing_dict["twoD_method"].append(twoD_method)
                    missing_dict["well_fov"].append(well_fov)
                    missing_dict["mask"].append(mask)
                    missing_dict["missing_mask_path"].append(mask_path)


# In[5]:


print(f"Total number of expected masks: {expected_count:,}")
print(f"Total number of masks found: {(expected_count - missing_count):,}")
print(f"Total number of missing masks: {missing_count:,}")


# In[6]:


df = pd.DataFrame(missing_dict)
df.sort_values(by=["patient_id", "well_fov", "twoD_method"], inplace=True)
df = (
    df.groupby(["patient_id", "well_fov", "twoD_method"])
    .size()
    .reset_index(name="missing_mask_count")
)
df.reset_index(drop=True, inplace=True)
df.head()


# In[7]:


# write to the loadfile
loadfile_path = pathlib.Path("../loadfiles/segmentation_loadfile.txt").resolve()
loadfile_path.parent.mkdir(parents=True, exist_ok=True)
with open(loadfile_path, "w") as f:
    for idx, row in df.iterrows():
        f.write(f"{row['patient_id']}\t{row['well_fov']}\t{row['twoD_method']}\n")


# In[8]:


df.groupby("patient_id").agg(
    {
        "well_fov": ["nunique"],
        "twoD_method": ["nunique"],
        "missing_mask_count": ["sum"],
    }
).reset_index()
