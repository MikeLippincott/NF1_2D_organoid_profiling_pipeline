#!/usr/bin/env python
# coding: utf-8

# # Process single cell and organoid profiles

# ## Import libraries

# In[1]:


import os
import pathlib
import pprint

import pandas as pd
from image_analysis_2D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_2D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import aggregate, annotate, feature_select, normalize
from pycytominer.cyto_utils import infer_cp_features

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


if not in_notebook:
    args_dict = parse_args()
    patient = args_dict["patient"]

    check_for_missing_args(
        patient=patient,
    )
else:
    print("Running in a notebook")
    patient = "SARCO219_T2"


# ## Set paths and variables

# In[3]:


# output path for single-cell profiles
output_dir = pathlib.Path(f"../../data/{patient}")
output_dir.mkdir(parents=True, exist_ok=True)
# operations to perform for feature selection
feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "blocklist",
    "drop_na_columns",
]


# ## Set dictionary with plates to process

# In[4]:


# create plate info dictionary
plate_info_dictionary = {
    "sc_max_projected": {
        "input_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/max_projected_sc.parquet"
        ).resolve(strict=True),
        "annotated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/4.annotated/max_projected_sc.parquet"
        ).resolve(),
        "normalized_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/5.normalized/max_projected_sc.parquet"
        ).resolve(),
        "feature_selected_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/6.feature_selected/max_projected_sc.parquet"
        ).resolve(),
        "aggregated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/7.aggregated/max_projected_sc.parquet"
        ).resolve(),
    },
    "sc_middle_slice": {
        "input_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/middle_slice_sc.parquet"
        ).resolve(strict=True),
        "annotated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/4.annotated/middle_slice_sc.parquet"
        ).resolve(),
        "normalized_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/5.normalized/middle_slice_sc.parquet"
        ).resolve(),
        "feature_selected_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/6.feature_selected/middle_slice_sc.parquet"
        ).resolve(),
        "aggregated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/7.aggregated/middle_slice_sc.parquet"
        ).resolve(),
    },
    "sc_middle_n_slice": {
        "input_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/middle_n_slice_sc.parquet"
        ).resolve(strict=True),
        "annotated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/4.annotated/middle_n_slice_sc.parquet"
        ).resolve(),
        "normalized_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/5.normalized/middle_n_slice_sc.parquet"
        ).resolve(),
        "feature_selected_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/6.feature_selected/middle_n_slice_sc.parquet"
        ).resolve(),
        "aggregated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/7.aggregated/middle_n_slice_sc.parquet"
        ).resolve(),
    },
    "organoid_max_projected": {
        "input_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/max_projected_organoid.parquet"
        ).resolve(strict=True),
        "annotated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/4.annotated/max_projected_organoid.parquet"
        ).resolve(),
        "normalized_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/5.normalized/max_projected_organoid.parquet"
        ).resolve(),
        "feature_selected_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/6.feature_selected/max_projected_organoid.parquet"
        ).resolve(),
        "aggregated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/7.aggregated/max_projected_organoid.parquet"
        ).resolve(),
    },
    "organoid_middle_slice": {
        "input_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/middle_slice_organoid.parquet"
        ).resolve(strict=True),
        "annotated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/4.annotated/middle_slice_organoid.parquet"
        ).resolve(),
        "normalized_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/5.normalized/middle_slice_organoid.parquet"
        ).resolve(),
        "feature_selected_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/6.feature_selected/middle_slice_organoid.parquet"
        ).resolve(),
        "aggregated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/7.aggregated/middle_slice_organoid.parquet"
        ).resolve(),
    },
    "organoid_middle_n_slice": {
        "input_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/middle_n_slice_organoid.parquet"
        ).resolve(strict=True),
        "annotated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/4.annotated/middle_n_slice_organoid.parquet"
        ).resolve(),
        "normalized_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/5.normalized/middle_n_slice_organoid.parquet"
        ).resolve(),
        "feature_selected_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/6.feature_selected/middle_n_slice_organoid.parquet"
        ).resolve(),
        "aggregated_path": pathlib.Path(
            f"{image_base_dir}/data/{patient}/2D_analysis/7.aggregated/middle_n_slice_organoid.parquet"
        ).resolve(),
    },
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Process data with pycytominer

# In[5]:


drug_information_df = pd.read_csv(
    pathlib.Path(f"{root_dir}/4.preprocess_features/data/drugs/drug_information.csv")
)


# In[6]:


drug_information_df = pd.read_csv(
    pathlib.Path(f"{root_dir}/4.preprocess_features/data/drugs/drug_information.csv")
)


# In[7]:


drug_information_df = pd.read_csv(
    pathlib.Path(f"{root_dir}/4.preprocess_features/data/drugs/drug_information.csv")
)


# In[8]:


for plate, info in tqdm.tqdm(plate_info_dictionary.items()):
    print(f"Performing pycytominer pipeline for {plate}")
    # make the parent directories for the output files
    for key, value in info.items():
        value.parent.mkdir(parents=True, exist_ok=True)

    profile_df = pd.read_parquet(info["input_path"])
    print(f"Profile shape: {profile_df.shape}")
    platemap_df = pd.read_csv(
        pathlib.Path(f"{root_dir}/data/{patient}/platemap/platemap.csv")
    )
    platemap_df.rename(columns={"unit": "Metadata_dose_unit"}, inplace=True)
    platemap_df.rename(columns={"well_position": "Metadata_Well"}, inplace=True)
    # Step 1: Annotation
    print("Performing annotation...")
    platemap_df = platemap_df.merge(
        drug_information_df, how="left", left_on="treatment", right_on="treatment"
    )
    annotate(
        profiles=profile_df,
        platemap=platemap_df,
        join_on=["Metadata_Well", "Metadata_Well"],
        output_file=info["annotated_path"],
        output_type="parquet",
    )

    # Load the annotated parquet file to fix metadata columns names
    annotated_df = pd.read_parquet(info["annotated_path"])

    [
        annotated_df.drop(columns=col, inplace=True)
        for col in annotated_df.columns
        if "plate" in col.lower()
    ]

    print("Performing normalization...")
    # Step 2: Normalization
    # Find the cp features based on the mask name or image
    if "organoid" in plate.lower():
        compartments = ["Organoid"]
    else:
        compartments = ["Cells", "Nuclei", "Cytoplasm"]
    cp_features = infer_cp_features(
        population_df=annotated_df, compartments=compartments
    )

    # Find the metadata features
    meta_features = infer_cp_features(
        population_df=annotated_df, compartments=compartments, metadata=True
    )

    # Perform normalization
    normalize(
        profiles=annotated_df,
        features=cp_features,
        meta_features=meta_features,
        method="standardize",
        output_file=info["normalized_path"],
        output_type="parquet",
    )

    print("Performing feature selection for...")

    # Step 3: Feature selection
    fs_df = feature_select(
        profiles=str(info["normalized_path"]),
        operation=feature_select_ops,
        features=cp_features,
        output_file=str(info["feature_selected_path"]),
        output_type="parquet",
    )

    cp_features = infer_cp_features(
        population_df=pd.read_parquet(info["feature_selected_path"]),
        compartments=compartments,
    )
    fs_df = pd.read_parquet(info["feature_selected_path"])
    # Step 4: Aggregation
    print("Performing aggregation...")
    single_cells_per_well = (
        fs_df.groupby("Metadata_Well").size().reset_index(name="Metadata_cell_count")
    )
    # merge back into the fs_df
    fs_df = fs_df.merge(single_cells_per_well, on="Metadata_Well", how="left")
    aggregate(
        population_df=fs_df,
        strata=["Metadata_treatment", "Metadata_dose"],
        features=cp_features,
        operation="median",
        output_file=str(info["aggregated_path"]),
        output_type="parquet",
    )
    print(f"Aggregation has been performed for:\n{plate}")


# In[14]:


df = pd.read_parquet(info["normalized_path"])
[x for x in df.columns if "metadata" in x.lower()]
