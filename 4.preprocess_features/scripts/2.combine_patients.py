#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_2D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import aggregate, feature_select

root_dir, in_notebook = init_notebook()
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


patient_ids_path = pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(patient_ids_path, header=None, names=["patient_id"], dtype=str)[
    "patient_id"
].to_list()

all_patients_output_path = pathlib.Path(
    f"{profile_base_dir}/data/all_patient_profiles"
).resolve()
all_patients_output_path.mkdir(parents=True, exist_ok=True)


# In[3]:


levels_to_merge_dict = {
    "max_projection": {
        "sc": [],
        "organoid": [],
    },
    "middle_slice": {
        "sc": [],
        "organoid": [],
    },
    "middle_n_slice": {
        "sc": [],
        "organoid": [],
    },
}


# In[4]:


for patient in patients:
    norm_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/2D_analysis/5.normalized"
    )
    for file in norm_path.glob("*.parquet"):
        if "max_projected" in file.name and "sc" in file.name:
            levels_to_merge_dict["max_projection"]["sc"].append(file)
        elif "max_projected" in file.name and "organoid" in file.name:
            levels_to_merge_dict["max_projection"]["organoid"].append(file)
        elif "middle_slice" in file.name and "sc" in file.name:
            levels_to_merge_dict["middle_slice"]["sc"].append(file)
        elif "middle_slice" in file.name and "organoid" in file.name:
            levels_to_merge_dict["middle_slice"]["organoid"].append(file)
        elif "middle_n_slice" in file.name and "sc" in file.name:
            levels_to_merge_dict["middle_n_slice"]["sc"].append(file)
        elif "middle_n_slice" in file.name and "organoid" in file.name:
            levels_to_merge_dict["middle_n_slice"]["organoid"].append(file)
        else:
            raise ValueError(f"File {file} does not match expected naming scheme.")


# In[5]:


feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    # "variance_threshold", # comment out to remove variance thresholding
    # "correlation_threshold", # comment out to remove correlation thresholding
]
na_cutoff = 0.05
corr_threshold = 0.9
freq_cut = 0.01
unique_cut = 0.01


# In[6]:


df = pd.read_parquet(levels_to_merge_dict["middle_slice"]["sc"][0])
sc_blocklist = [
    x
    for x in df.columns
    if "area" in x.lower()
    and (
        "max" in x.lower()
        or "min" in x.lower()
        or "bbox" in x.lower()
        or "center" in x.lower()
    )
]
sc_blocklist += [x for x in df.columns if "boundingbox" in x.lower()]
# write the blocklist to a file
# add "blocklist" the beginning of the list
sc_blocklist = ["blocklist"] + sc_blocklist
sc_blocklist_path = pathlib.Path(
    f"{root_dir}/4.preprocess_features/data/blocklist/sc_blocklist.txt"
).resolve()
sc_blocklist_path.parent.mkdir(parents=True, exist_ok=True)
with open(sc_blocklist_path, "w") as f:
    for item in sc_blocklist:
        f.write(f"{item}\n")

# organoid blocklist
df = pd.read_parquet(levels_to_merge_dict["middle_slice"]["organoid"][0])
organoid_blocklist = [
    x
    for x in df.columns
    if "area" in x.lower()
    and (
        "max" in x.lower()
        or " min" in x.lower()
        or "bbox" in x.lower()
        or "center" in x.lower()
    )
]
organoid_blocklist += [x for x in df.columns if "boundingbox" in x.lower()]
# write the blocklist to a file
# add "blocklist" the beginning of the list
organoid_blocklist = ["blocklist"] + organoid_blocklist
organoid_blocklist_path = pathlib.Path(
    f"{root_dir}/4.preprocess_features/data/blocklist/organoid_blocklist.txt"
).resolve()
organoid_blocklist_path.parent.mkdir(parents=True, exist_ok=True)
with open(organoid_blocklist_path, "w") as f:
    for item in organoid_blocklist:
        f.write(f"{item}\n")


# In[ ]:


dict_of_dfs_to_process = {
    "middle_slice": {
        "sc": [],
        "organoid": [],
    },
    "max_projection": {
        "sc": [],
        "organoid": [],
    },
    "middle_n_slice": {
        "sc": [],
        "organoid": [],
    },
}
for profile_type in levels_to_merge_dict.keys():
    for compartment in levels_to_merge_dict[profile_type].keys():
        list_of_dfs = []
        for file in levels_to_merge_dict[profile_type][compartment]:
            patient_id = str(file.parent).split("/")[-3]  # -3 is the patient ID
            df = pd.read_parquet(file)
            df["Metadata_patient_tumor"] = patient_id
            list_of_dfs.append(df)
        df = pd.concat(list_of_dfs, ignore_index=True)
        new_df_path = pathlib.Path(
            f"{all_patients_output_path}/{profile_type}/{compartment}_profiles.parquet"
        ).resolve()
        new_df_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            new_df_path,
            index=False,
        )
        dict_of_dfs_to_process[profile_type][compartment] = new_df_path


# In[8]:


for profile_type in dict_of_dfs_to_process.keys():
    for compartment in dict_of_dfs_to_process[profile_type].keys():
        df = pd.read_parquet(dict_of_dfs_to_process[profile_type][compartment])
        print(f"DataFrame for {profile_type} {compartment} has the shape: {df.shape}")
        df["Metadata_patient"] = df["Metadata_patient_tumor"].str.split("_").str[0]
        df["Metadata_tumor"] = df["Metadata_patient_tumor"].str.split("_").str[1]
        if compartment == "sc":
            blocklist_path = pathlib.Path(
                f"{root_dir}/4.preprocess_features/data/blocklist/sc_blocklist.txt"
            )
            metadata_cols = [x for x in df.columns if "Metadata_" in x]
            # only perform feature selection on DMSO and staurosporine treatments and apply to rest of profiles
            all_trt_df = df.copy()
            df = df.loc[df["Metadata_treatment"].isin(["DMSO", "Staurosporine"])]
            # feature selection
            feature_columns = [col for col in df.columns if col not in metadata_cols]
            features_df = df[feature_columns]
            fs_profiles = feature_select(
                features_df,
                operation=feature_select_ops,
                features=feature_columns,
                na_cutoff=na_cutoff,
                # corr_threshold=corr_threshold, # comment out to use default value
                # freq_cut=freq_cut, # comment out to use default value
                # unique_cut=unique_cut, # comment out to use default value
            )
            original_data_shape = features_df.shape
            # apply feature selection to all profiles
            fs_profiles = all_trt_df[
                [col for col in all_trt_df.columns if col in fs_profiles.columns]
            ]
            fs_profiles = pd.concat(
                [
                    all_trt_df[metadata_cols].reset_index(drop=True),
                    fs_profiles.reset_index(drop=True),
                ],
                axis=1,
            )
            fs_profiles.to_parquet(
                f"{all_patients_output_path}/{profile_type}/sc_fs_profiles.parquet",
                index=False,
            )
            feature_columns = [
                col for col in fs_profiles.columns if col not in metadata_cols
            ]
            features_df = fs_profiles[feature_columns]
            # aggregate the profiles
            sc_agg_df = aggregate(
                population_df=fs_profiles,
                strata=[
                    "Metadata_patient_tumor",
                    "Metadata_Well",
                    "Metadata_treatment",
                    "Metadata_dose",
                    "Metadata_dose_unit",
                    "Metadata_target",
                    "Metadata_class",
                    "Metadata_therapeutic_categories",
                ],
                features=feature_columns,
                operation="median",
            )
            sc_agg_df.to_parquet(
                f"{all_patients_output_path}/{profile_type}/sc_agg_profiles.parquet",
                index=False,
            )
            # consensus profiles
            sc_consensus_df = aggregate(
                population_df=fs_profiles,
                strata=[
                    "Metadata_patient_tumor",
                    "Metadata_treatment",
                    "Metadata_dose",
                    "Metadata_dose_unit",
                    "Metadata_target",
                    "Metadata_class",
                    "Metadata_therapeutic_categories",
                ],
                features=feature_columns,
                operation="median",
            )
            sc_consensus_df.to_parquet(
                f"{all_patients_output_path}/{profile_type}/sc_consensus_profiles.parquet",
                index=False,
            )
            print(
                "The number features before feature selection:", original_data_shape[1]
            )
            print("The number features after feature selection:", fs_profiles.shape[1])

        elif compartment == "organoid":
            blocklist_path = pathlib.Path(
                f"{root_dir}/4.processing_image_based_profiles/data/blocklist/organoid_blocklist.txt"
            )
            metadata_cols = [x for x in df.columns if "Metadata_" in x]

            all_trt_df = df.copy()
            df = df.loc[df["Metadata_treatment"].isin(["DMSO", "Staurosporine"])]
            feature_columns = [col for col in df.columns if col not in metadata_cols]
            features_df = df[feature_columns]
            fs_profiles = feature_select(
                features_df,
                operation=feature_select_ops,
                features=feature_columns,
                na_cutoff=na_cutoff,
                corr_threshold=corr_threshold,
                freq_cut=freq_cut,
                unique_cut=unique_cut,
            )
            fs_profiles = all_trt_df[
                [col for col in all_trt_df.columns if col in fs_profiles.columns]
            ]
            original_data_shape = features_df.shape
            fs_profiles = pd.concat(
                [
                    all_trt_df[metadata_cols].reset_index(drop=True),
                    fs_profiles.reset_index(drop=True),
                ],
                axis=1,
            )
            fs_profiles.to_parquet(
                f"{all_patients_output_path}/{profile_type}/organoid_fs_profiles.parquet",
                index=False,
            )
            feature_columns = [
                col for col in fs_profiles.columns if col not in metadata_cols
            ]
            features_df = fs_profiles[feature_columns]
            # aggregate the profiles
            agg_df = aggregate(
                population_df=fs_profiles,
                strata=[
                    "Metadata_patient_tumor",
                    "Metadata_Well",
                    "Metadata_treatment",
                    "Metadata_dose",
                    "Metadata_dose_unit",
                    "Metadata_target",
                    "Metadata_class",
                    "Metadata_therapeutic_categories",
                ],
                features=feature_columns,
                operation="median",
            )
            agg_df.to_parquet(
                f"{all_patients_output_path}/{profile_type}/organoid_agg_profiles.parquet",
                index=False,
            )
            # consensus profiles
            consensus_df = aggregate(
                population_df=fs_profiles,
                strata=[
                    "Metadata_patient_tumor",
                    "Metadata_treatment",
                    "Metadata_dose",
                    "Metadata_dose_unit",
                    "Metadata_target",
                    "Metadata_class",
                    "Metadata_therapeutic_categories",
                ],
                features=feature_columns,
                operation="median",
            )
            consensus_df.to_parquet(
                f"{all_patients_output_path}/{profile_type}/organoid_consensus_profiles.parquet",
                index=False,
            )

            print(
                "The number features before feature selection:", original_data_shape[1]
            )
            print("The number features after feature selection:", fs_profiles.shape[1])


# In[ ]:


import pandas as pd

pd.read_parquet("../../data/all_patient_profiles/max_projection/sc_fs_profiles.parquet")


# In[ ]:
