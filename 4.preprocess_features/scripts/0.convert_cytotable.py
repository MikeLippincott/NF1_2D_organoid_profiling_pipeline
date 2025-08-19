#!/usr/bin/env python
# coding: utf-8

# # Convert SQLite output(s) to parquet files with CytoTable

# ## Import libraries

# In[1]:


import argparse
import logging
import pathlib
import uuid

import duckdb
import pandas as pd
import tqdm

# cytotable will merge objects from SQLite file into single cells and save as parquet file
from cytotable import convert, presets
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

# Set the logging level to a higher level to avoid outputting unnecessary errors from config file in convert function
logging.getLogger().setLevel(logging.ERROR)
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[ ]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--patient",
        type=str,
        help="Patient ID",
    )

    args = parser.parse_args()
    patient = args.patient
else:
    print("Running in a notebook")
    patient = "NF0016"

middle_slice_input = pathlib.Path(
    f"../../data/{patient}/cellprofiler_middle_slice_output/"
).resolve(strict=True)
max_projected_input = pathlib.Path(
    f"../../data/{patient}/cellprofiler_zmax_proj_output/"
).resolve(strict=True)

# directory for processed data
output_dir = pathlib.Path(f"../../data/{patient}/0.converted/").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
middle_slice_sc_output = pathlib.Path(output_dir, "middle_slice_sc.parquet").resolve()
max_projected_sc_output = pathlib.Path(output_dir, "max_projected_sc.parquet").resolve()
middle_slice_organoid_output = pathlib.Path(
    output_dir, "middle_slice_organoid.parquet"
).resolve()
max_projected_organoid_output = pathlib.Path(
    output_dir, "max_projected_organoid.parquet"
).resolve()


# ## Set paths and variables

# In[3]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"

# update preset to include site metadata and cell counts
joins = presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"].replace(
    "Image_Metadata_Well,",
    "Image_Metadata_Well, Image_Metadata_Site, Image_Count_Cells,",
)

# type of file output from cytotable (currently only parquet)
dest_datatype = "parquet"


well_fov_dict = {}
for sqlite_dir in [middle_slice_input, max_projected_input]:
    twoD_type = sqlite_dir.name.split("_out")[0].split("cellprofiler_")[1]
    well_fov_dict[twoD_type] = {}
    sqlites = list(sqlite_dir.rglob("*sqlite"))
    sqlites.sort()  # sort to ensure consistent order
    for file_path in sqlites:
        well_fov = file_path.parent.stem
        well_fov_dict[twoD_type][well_fov] = {
            "image_path": file_path,
            "output_dir": output_dir / twoD_type / f"{well_fov}",
        }


# ## Convert SQLite to parquet file(s) for single-cell profiles

# In[4]:


output_dict_of_dfs = {}
for sqlite_dir in [middle_slice_input, max_projected_input]:
    output_dict_of_dfs[sqlite_dir.name.split("_out")[0].split("cellprofiler_")[1]] = {
        "df_list": [],
    }
output_dict_of_dfs


# In[5]:


total = 0
errors = 0
# loop through the middle and zmax projected sqlite files
for featurization_type in well_fov_dict.keys():
    for well_fov, file_info in tqdm.tqdm(well_fov_dict[featurization_type].items()):
        sqlite_file = file_info["image_path"]
        total += 1
        # convert the sqlite file to a single cell parquet file
        try:
            df = convert(
                sqlite_file,
                preset=preset,
                joins=joins,
                chunk_size=500,
                dest_datatype=dest_datatype,
                dest_path=f"{well_fov_dict[featurization_type][well_fov]['output_dir']}_sc.parquet",
                parsl_config=Config(
                    executors=[HighThroughputExecutor()],
                    run_dir=f"cytotable_runinfo/{uuid.uuid4().hex}",
                ),
            )
            output_dict_of_dfs[featurization_type]["df_list"].append(
                f"{well_fov_dict[featurization_type][well_fov]['output_dir']}_sc.parquet"
            )
        except Exception as e:
            errors += 1
            print(f"Error processing {sqlite_file}: {e}")
            continue
print(f"Total files processed: {total}")
print(f"Total errors encountered: {errors}")


# In[6]:


# read in the dataframes and concatenate them in place
for featurization_type in output_dict_of_dfs.keys():
    print(
        f"Concatenating {len(output_dict_of_dfs[featurization_type]['df_list'])} dataframes for {featurization_type}"
    )
    df_list = [
        pd.read_parquet(df) for df in output_dict_of_dfs[featurization_type]["df_list"]
    ]
    output_dict_of_dfs[featurization_type]["df"] = pd.concat(df_list, ignore_index=True)
    # Define the list of columns to prioritize and prefix
    prioritized_columns = [
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Cells_Location_Center_X",
        "Cells_Location_Center_Y",
        "Image_Count_Cells",
    ]

    # If any, drop rows where "Metadata_ImageNumber" is NaN (artifact of cytotable)
    output_dict_of_dfs[featurization_type]["df"] = output_dict_of_dfs[
        featurization_type
    ]["df"].dropna(subset=["Metadata_ImageNumber"])

    # Rearrange columns and add "Metadata" prefix in one line
    output_dict_of_dfs[featurization_type]["df"] = output_dict_of_dfs[
        featurization_type
    ]["df"][
        prioritized_columns
        + [
            col
            for col in output_dict_of_dfs[featurization_type]["df"].columns
            if col not in prioritized_columns
        ]
    ].rename(
        columns=lambda col: "Metadata_" + col if col in prioritized_columns else col
    )
    # rename Image_Metadata_Well
    output_dict_of_dfs[featurization_type]["df"] = output_dict_of_dfs[
        featurization_type
    ]["df"].rename(columns={"Image_Metadata_Well": "Metadata_Well"})

    if featurization_type == "middle_slice":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            middle_slice_sc_output, index=False
        )
    elif featurization_type == "zmax_proj":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            max_projected_sc_output, index=False
        )
    print(
        f"Saved {featurization_type} data to {output_dict_of_dfs[featurization_type]['df'].shape[0]} rows in {output_dict_of_dfs[featurization_type]['df'].shape[1]} columns"
    )


# ## Extract organoid only profiles

# In[7]:


output_dict_of_dfs = {}
for sqlite_dir in [middle_slice_input, max_projected_input]:
    output_dict_of_dfs[sqlite_dir.name.split("_out")[0].split("cellprofiler_")[1]] = {
        "df_list": [],
    }
output_dict_of_dfs


# In[8]:


total = 0
errors = 0
for featurization_type in well_fov_dict.keys():
    print(f"Processing {featurization_type} files")
    for well_fov, file_info in tqdm.tqdm(well_fov_dict[featurization_type].items()):
        well = well_fov.split("-")[0]
        fov = well_fov.split("-")[1]
        sqlite_file = file_info["image_path"]
        total += 1
        try:
            # Create a DuckDB connection
            with duckdb.connect(sqlite_file) as con:
                # get the organoid table
                organoid_table = con.execute("SELECT * FROM Per_Organoid").df()
                organoid_table.rename(
                    columns={
                        "ImageNumber": "Metadata_ImageNumber",
                        "Organoid_Number_Object_Number": "Metadata_Organoid_Number_Object_Number",
                        "Image_Metadata_Well": "Metadata_Well",
                    },
                    inplace=True,
                )
                organoid_table.insert(0, "Metadata_Well_FOV", well_fov)
                organoid_table.insert(1, "Metadata_FOV", fov)
                organoid_table.insert(2, "Metadata_Well", well)
            output_dict_of_dfs[featurization_type]["df_list"].append(organoid_table)

        except Exception as e:
            errors += 1
            print(f"Error processing {sqlite_file}: {e}")
            continue


# In[ ]:


# read in the dataframes and concatenate them in place
for featurization_type in output_dict_of_dfs.keys():
    print(
        f"Concatenating {len(output_dict_of_dfs[featurization_type]['df_list'])} dataframes for {featurization_type}"
    )
    output_dict_of_dfs[featurization_type]["df"] = pd.concat(
        output_dict_of_dfs[featurization_type]["df_list"], ignore_index=True
    )

    # If any, drop rows where "Metadata_ImageNumber" is NaN (artifact of cytotable)
    output_dict_of_dfs[featurization_type]["df"] = output_dict_of_dfs[
        featurization_type
    ]["df"].dropna(subset=["Metadata_ImageNumber"])
    if featurization_type == "middle_slice":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            middle_slice_organoid_output, index=False
        )
    elif featurization_type == "zmax_proj":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            max_projected_organoid_output, index=False
        )
    print(
        f"Saved {featurization_type} data to {output_dict_of_dfs[featurization_type]['df'].shape[0]} rows in {output_dict_of_dfs[featurization_type]['df'].shape[1]} columns"
    )
