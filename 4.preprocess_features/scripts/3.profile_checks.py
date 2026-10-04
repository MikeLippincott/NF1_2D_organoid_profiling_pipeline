#!/usr/bin/env python
# coding: utf-8

# # Checking profiles prior to sending off for analysis.
# This step is to check that the profiles look as expected before sending them off for analysis.
# This includes checking that the profiles have the expected number of features, that the features are named as expected, and that there are no missing values in the profiles.
# We run this in a notebook without assertions as we want to be able to investegate where any issues are if they arise.

# In[1]:


import os
import pathlib

import duckdb
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
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


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


patient_profile_path_dict = {}
for patient in tqdm.tqdm(patients, desc="Patients", leave=True):
    profile_path_dict = {
        "sc": {
            "normalized": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/{patient}/2D_analysis/5.normalized/"
                )
                .resolve()
                .glob("*")
                if x.is_file() and x.suffix == ".parquet" and "sc" in x.stem
            ],
            "feature_selected": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/{patient}/2D_analysis/6.feature_selected/"
                )
                .resolve()
                .glob("*")
                if x.is_file() and x.suffix == ".parquet" and "sc" in x.stem
            ],
            "aggregated": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/{patient}/2D_analysis/7.aggregated/"
                )
                .resolve()
                .glob("*")
                if x.is_file() and x.suffix == ".parquet" and "sc" in x.stem
            ],
        },
        "organoid": {
            "normalized": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/{patient}/2D_analysis/5.normalized/"
                )
                .resolve()
                .glob("*")
                if x.is_file() and x.suffix == ".parquet" and "organoid" in x.stem
            ],
            "feature_selected": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/{patient}/2D_analysis/6.feature_selected/"
                )
                .resolve()
                .glob("*")
                if x.is_file() and x.suffix == ".parquet" and "organoid" in x.stem
            ],
            "aggregated": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/{patient}/2D_analysis/7.aggregated/"
                )
                .resolve()
                .glob("*")
                if x.is_file() and x.suffix == ".parquet" and "organoid" in x.stem
            ],
        },
    }
    patient_profile_path_dict[patient] = profile_path_dict


# ## Perform checks on profiles

# In[4]:


# first up: ensure that all profiles are actually present
# there should be six profiles per profile type
# profile types are:
# sc_normalized, sc_feature_selected, sc_aggregated,
# organoid_normalized, organoid_feature_selected, organoid_aggregated
for patient, profile_path_dict in tqdm.tqdm(
    patient_profile_path_dict.items(), desc="Patients", leave=True
):
    for profile_type, profile_type_dict in tqdm.tqdm(
        profile_path_dict.items(), desc=f"{patient} compartments", leave=False
    ):
        for profile_subtype, profile_paths in tqdm.tqdm(
            profile_type_dict.items(),
            desc=f"{patient} {profile_type} profile subtypes",
            leave=False,
        ):
            if len(profile_paths) != 3:
                print(
                    f"Patient {patient} has {len(profile_paths)} profiles for {profile_type} {profile_subtype}, expected 3."
                )


# ### Checking profile shape and missing value numbers

# In[5]:


# profile_info_list = []
# for patient, profile_path_dict in tqdm.tqdm(
#     patient_profile_path_dict.items(), desc="Patients", leave=True
# ):
#     # check that all profiles have the same number of features and that the features are named as expected.
#     for compartment, profile_paths_sub_dict in tqdm.tqdm(
#         profile_path_dict.items(), desc=f"{patient} compartments", leave=False
#     ):
#         for profile_type, profile_paths in tqdm.tqdm(
#             profile_paths_sub_dict.items(),
#             desc=f"{patient} {compartment} profile types",
#             leave=False,
#         ):
#             for profile_path in tqdm.tqdm(
#                 profile_paths,
#                 desc=f"{patient} {compartment} {profile_type} profiles",
#                 leave=False,
#             ):
#                 # projection method is all but the last part of the filename, which is the profile type (normalized, feature_selected, aggregated)
#                 projection_method = ("_").join(profile_path.stem.split("_")[:-1])
#                 profile_df = pd.read_parquet(profile_path)
#                 profile_info_list.append(
#                     {
#                         "patient": patient,
#                         "compartment": compartment,
#                         "profile_type": profile_type,
#                         "projection_method": projection_method,
#                         "num_features": profile_df.shape[1],
#                         "missing_values": profile_df.isnull().sum().sum(),
#                     }
#                 )

# profile_info_df = pd.DataFrame(profile_info_list)
# profile_info_df


# Noting here that primarily the normalized profiles show up with NAs leftover.
# This confirms that the feature selection process actually removed these features from the profiles, and that the NAs are not a result of an issue with the feature selection process.

# ### Validating no duplicated features in profiles

# In[6]:


# duplicates = 0
# for patient, profile_path_dict in tqdm.tqdm(
#     patient_profile_path_dict.items(), desc="Patients", leave=True
# ):
#     # check for duplicated features in the profiles.
#     for compartment, profile_paths_sub_dict in tqdm.tqdm(
#         profile_path_dict.items(), desc=f"{patient} compartments", leave=False
#     ):
#         for profile_type, profile_paths in tqdm.tqdm(
#             profile_paths_sub_dict.items(),
#             desc=f"{patient} {compartment} profile types",
#             leave=False,
#         ):
#             for profile_path in tqdm.tqdm(
#                 profile_paths,
#                 desc=f"{patient} {compartment} {profile_type} profiles",
#                 leave=False,
#             ):
#                 # projection method is all but the last part of the filename, which is the profile type (normalized, feature_selected, aggregated)
#                 projection_method = ("_").join(profile_path.stem.split("_")[:-1])
#                 profile_df = pd.read_parquet(profile_path)
#                 duplicated_features = profile_df.columns[
#                     profile_df.columns.duplicated()
#                 ].tolist()
#                 if len(duplicated_features) > 0:
#                     duplicates += 1
#                     print(f"\n================{patient} profiles================\n")
#                     print(f"\n----------------{compartment} profiles----------------\n")
#                     print(
#                         f"{profile_type} - {compartment} - {projection_method}: duplicated features: {duplicated_features}"
#                     )

# print(f"Total profiles with duplicated features: {duplicates}")


# In[7]:


# duplicates = 0
# for patient, profile_path_dict in tqdm.tqdm(
#     patient_profile_path_dict.items(), desc="Patients", leave=True
# ):
#     # check for features that end in _x or _y, which may indicate that there are duplicated features in the profiles.
#     for compartment, profile_paths_sub_dict in tqdm.tqdm(
#         profile_path_dict.items(), desc=f"{patient} compartments", leave=False
#     ):
#         for profile_type, profile_paths in tqdm.tqdm(
#             profile_paths_sub_dict.items(),
#             desc=f"{patient} {compartment} profile types",
#             leave=False,
#         ):
#             for profile_path in tqdm.tqdm(
#                 profile_paths,
#                 desc=f"{patient} {compartment} {profile_type} profiles",
#                 leave=False,
#             ):
#                 # projection method is all but the last part of the filename, which is the profile type (normalized, feature_selected, aggregated)
#                 projection_method = ("_").join(profile_path.stem.split("_")[:-1])
#                 profile_df = pd.read_parquet(profile_path)
#                 features_ending_in_x = [
#                     col for col in profile_df.columns if col.endswith("_x")
#                 ]
#                 features_ending_in_y = [
#                     col for col in profile_df.columns if col.endswith("_y")
#                 ]
#                 if len(features_ending_in_x) > 0 or len(features_ending_in_y) > 0:
#                     duplicates += 1
#                     print(f"\n================{patient} profiles================\n")
#                     print(f"\n----------------{compartment} profiles----------------\n")
#                     print(
#                         f"{profile_type} - {compartment} - {projection_method}: features ending in _x: {features_ending_in_x}"
#                     )

# print(f"Total profiles with features ending in _x or _y: {duplicates}")


# ### Get the stats and checks on the combined profiles also

# In[8]:


combined_profiles_dict = {}
projection_methods = ["max_projection", "middle_slice", "middle_n_slice"]
for projection_method in tqdm.tqdm(
    projection_methods, desc="Projection methods", leave=True
):
    combined_profiles_sub_dict = {
        "sc": {
            "normalized": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "sc" in x.stem
                and "_fs_" not in x.stem
                and "_agg" not in x.stem
                and "consensus" not in x.stem
            ],
            "feature_selected": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "sc" in x.stem
                and "_fs_" in x.stem
            ],
            "aggregated": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "sc" in x.stem
                and "_agg" in x.stem
            ],
            "consensus": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "consensus" in x.stem
                and "sc" in x.stem
            ],
        },
        "organoid": {
            "normalized": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "organoid" in x.stem
                and "_fs_" not in x.stem
                and "_agg" not in x.stem
                and "consensus" not in x.stem
            ],
            "feature_selected": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "organoid" in x.stem
                and "_fs_" in x.stem
            ],
            "aggregated": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "organoid" in x.stem
                and "_agg" in x.stem
            ],
            "consensus": [
                x
                for x in pathlib.Path(
                    f"{profile_base_dir}/data/all_patient_profiles/{projection_method}/"
                )
                .resolve()
                .glob("*")
                if x.is_file()
                and x.suffix == ".parquet"
                and "organoid" in x.stem
                and "consensus" in x.stem
            ],
        },
    }
    combined_profiles_dict[projection_method] = combined_profiles_sub_dict


# In[9]:


combined_profiles_list = []
for projection_method, combined_profiles_sub_dict in tqdm.tqdm(
    combined_profiles_dict.items(), desc="Projection methods", leave=True
):
    for compartment, profile_paths_sub_dict in tqdm.tqdm(
        combined_profiles_sub_dict.items(),
        desc=f"{projection_method} compartments",
        leave=False,
    ):
        for profile_type, profile_paths in tqdm.tqdm(
            profile_paths_sub_dict.items(),
            desc=f"{projection_method} {compartment} profile types",
            leave=False,
        ):
            if len(profile_paths) == 0:
                combined_profiles_list.append(
                    {
                        "projection_method": projection_method,
                        "compartment": compartment,
                        "profile_type": profile_type,
                        "profile_path": None,
                        "shape": None,
                        "missing_values": None,
                        "status": "missing",
                    }
                )
            else:
                for profile_path in tqdm.tqdm(
                    profile_paths,
                    desc=f"{projection_method} {compartment} {profile_type} profiles",
                    leave=False,
                ):
                    profile_df = pd.read_parquet(profile_path)
                    combined_profiles_list.append(
                        {
                            "projection_method": projection_method,
                            "compartment": compartment,
                            "profile_type": profile_type,
                            "profile_path": str(profile_path),
                            "shape": profile_df.shape,
                            "missing_values": int(profile_df.isnull().sum().sum()),
                            "status": "ok",
                        }
                    )

combined_profiles_df = pd.DataFrame(combined_profiles_list)
combined_profiles_df


# In[11]:


# search for faulty metadata values in the profiles, such as missing patient IDs or treatment information.
# count the number of missing values in the metadata columns of the profiles, such as Metadata_patient_tumor and Metadata_treatment. This can help identify if there are any profiles with missing metadata information that may need to be addressed before downstream analysis.
for profile_path in tqdm.tqdm(
    combined_profiles_df["profile_path"].dropna(),
    desc=f"Checking profiles for missing metadata values",
    leave=True,
):
    profile_df = pd.read_parquet(profile_path)
    metadata_columns = [col for col in profile_df.columns if "Metadata" in col]
    for metadata_column in metadata_columns:
        num_missing = profile_df[metadata_column].isnull().sum()
        if num_missing > 0:
            print(
                f"Profile {profile_path} has {num_missing} missing values in metadata column {metadata_column}."
            )


# In[5]:


import pathlib

import pandas as pd

df = pd.read_parquet(
    pathlib.Path(
        "/home/lippincm/mnt/bandicoot/NF1_organoid_data/data/NF0037_T1/2D_analysis/5.normalized/max_projected_sc.parquet"
    )
)
df


# In[ ]:
