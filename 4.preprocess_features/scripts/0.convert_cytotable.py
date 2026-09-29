#!/usr/bin/env python
# coding: utf-8

# # Convert SQLite output(s) to parquet files with CytoTable

# ## Import libraries

# In[1]:


import logging
import os
import pathlib
import shutil
import sqlite3
import uuid

import duckdb
import pandas as pd
import tqdm

# cytotable will merge objects from SQLite file into single cells and save as parquet file
from cytotable import convert, presets
from image_analysis_2D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_2D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

# Set the logging level to a higher level to avoid outputting unnecessary errors from config file in convert function
logging.getLogger().setLevel(logging.ERROR)
root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
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
    patient = "SARCO361_T1"


max_projected_input = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/2a.cellprofiler_zmax_proj_output/"
).resolve(strict=True)
middle_slice_input = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/2b.cellprofiler_middle_slice_output/"
).resolve(strict=True)
middle_n_slice_input = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/2c.cellprofiler_middle_n_slice_max_proj_output/"
).resolve(strict=True)

# directory for processed data
output_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/"
).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

max_projected_sc_output = pathlib.Path(output_dir, "max_projected_sc.parquet").resolve()
middle_slice_sc_output = pathlib.Path(output_dir, "middle_slice_sc.parquet").resolve()
middle_n_slice_sc_output = pathlib.Path(
    output_dir, "middle_n_slice_sc.parquet"
).resolve()

max_projected_organoid_output = pathlib.Path(
    output_dir, "max_projected_organoid.parquet"
).resolve()
middle_slice_organoid_output = pathlib.Path(
    output_dir, "middle_slice_organoid.parquet"
).resolve()
middle_n_slice_organoid_output = pathlib.Path(
    output_dir, "middle_n_slice_organoid.parquet"
).resolve()


# In[3]:


full_schema_for_sc = pd.read_parquet(
    pathlib.Path(
        f"{image_base_dir}/data/NF0014_T1/2D_analysis/3.converted/zmax_proj/C2-1_sc.parquet"
    )
).head(0)
full_schema_for_organoid = pd.read_parquet(
    pathlib.Path(
        f"{image_base_dir}/data/NF0014_T1/2D_analysis/3.converted/max_projected_organoid.parquet"
    )
).head(0)


# ## Set paths and variables

# In[4]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"

# update preset to include site metadata and cell counts
joins = presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"]

presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"] = joins

# type of file output from cytotable (currently only parquet)
dest_datatype = "parquet"


well_fov_dict = {}
for sqlite_dir in [max_projected_input, middle_slice_input, middle_n_slice_input]:
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

# In[5]:


output_dict_of_dfs = {}
for sqlite_dir in [max_projected_input, middle_slice_input, middle_n_slice_input]:
    print(sqlite_dir)
    output_dict_of_dfs[sqlite_dir.name.split("_out")[0].split("cellprofiler_")[1]] = {
        "df_list": [],
    }
output_dict_of_dfs


# In[ ]:


total = 0
errors = 0
already_converted_skips = 0
empty_sqlite_skips = 0

# loop through the middle and zmax projected sqlite files
for featurization_type in tqdm.tqdm(well_fov_dict.keys(), leave=True):
    for well_fov, file_info in tqdm.tqdm(
        well_fov_dict[featurization_type].items(), leave=False
    ):
        sqlite_file = file_info["image_path"]
        total += 1

        # Check if all three measurement tables are empty before running CytoTable
        tables_to_check = ["Per_Cells", "Per_Cytoplasm", "Per_Nuclei"]
        all_three_empty = True
        try:
            with sqlite3.connect(sqlite_file) as con:
                for table_name in tables_to_check:
                    table_exists = con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table_name,),
                    ).fetchone()

                    # If table exists and has at least one row, this file is not empty
                    if table_exists:
                        has_rows = con.execute(
                            f"SELECT EXISTS(SELECT 1 FROM {table_name} LIMIT 1)"
                        ).fetchone()[0]
                        if has_rows:
                            all_three_empty = False
                            break
        except sqlite3.Error as e:
            errors += 1
            print(f"SQLite read error for {sqlite_file}: {e}")
            continue

        if all_three_empty:
            empty_sqlite_skips += 1
            continue

        # convert the sqlite file to a single cell parquet file
        run_info_dir = pathlib.Path(f"cytotable_runinfo/{uuid.uuid4().hex}").resolve()
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
                    run_dir=str(run_info_dir),
                ),
            )
            output_dict_of_dfs[featurization_type]["df_list"].append(
                f"{well_fov_dict[featurization_type][well_fov]['output_dir']}_sc.parquet"
            )

            # remove the temporary run info directory
            if run_info_dir.exists():
                shutil.rmtree(run_info_dir)

        except Exception as e:
            if not "An existing file or directory was provided as dest_path" in str(e):
                errors += 1
                print(f"Error processing {sqlite_file}: {e}")
                # write an empty parquet file to use down the line
                full_schema_for_sc.to_parquet(
                    f"{well_fov_dict[featurization_type][well_fov]['output_dir']}_sc.parquet"
                )
            else:
                already_converted_skips += 1
            if run_info_dir.exists():
                shutil.rmtree(run_info_dir)
            continue

# remove any straggling run info directories
cytotable_runinfo_dir = pathlib.Path("cytotable_runinfo")
if cytotable_runinfo_dir.exists():
    shutil.rmtree(cytotable_runinfo_dir)

print(f"Total files processed: {total}")
print(f"Total files already converted and skipped: {already_converted_skips}")
print(f"Total errors encountered: {errors}")
print(f"Total empty sqlite files skipped: {empty_sqlite_skips}")


# In[7]:


output_dict_of_dfs = {
    "zmax_proj": {
        "df_list": [
            x
            for x in pathlib.Path(
                f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/zmax_proj/"
            ).rglob("*.parquet")
        ]
    },
    "middle_slice": {
        "df_list": [
            x
            for x in pathlib.Path(
                f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/middle_slice/"
            ).rglob("*.parquet")
        ]
    },
    "middle_n_slice": {
        "df_list": [
            x
            for x in pathlib.Path(
                f"{image_base_dir}/data/{patient}/2D_analysis/3.converted/middle_n_slice_max_proj/"
            ).glob("*.parquet")
        ]
    },
}


# In[8]:


# read in the dataframes and concatenate them in place
for featurization_type in tqdm.tqdm(
    output_dict_of_dfs.keys(), leave=True, desc="Concatenating dataframes"
):
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
    if featurization_type == "zmax_proj":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            max_projected_sc_output, index=False
        )
    elif featurization_type == "middle_slice":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            middle_slice_sc_output, index=False
        )
    elif featurization_type == "middle_n_slice":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            middle_n_slice_sc_output, index=False
        )
    print(
        f"Saved {featurization_type} data to {output_dict_of_dfs[featurization_type]['df'].shape[0]} rows in {output_dict_of_dfs[featurization_type]['df'].shape[1]} columns"
    )


# ## Extract organoid only profiles

# ## Set paths and variables

# In[9]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"

# update preset to include site metadata and cell counts
joins = presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"].replace(
    "Image_Metadata_Well,",
    "Image_Metadata_Well, Image_Metadata_Site",
)

# type of file output from cytotable (currently only parquet)
dest_datatype = "parquet"


well_fov_dict = {}
for sqlite_dir in tqdm.tqdm(
    [max_projected_input, middle_slice_input, middle_n_slice_input],
    desc="Processing sqlite directories",
):
    twoD_type = sqlite_dir.name.split("_out")[0].split("cellprofiler_")[1]
    well_fov_dict[twoD_type] = {}
    sqlites = list(sqlite_dir.rglob("*organoid*"))
    sqlites.sort()  # sort to ensure consistent order
    for file_path in tqdm.tqdm(sqlites, leave=False, desc="Processing sqlite files"):
        well_fov = file_path.parent.stem
        well_fov_dict[twoD_type][well_fov] = {
            "sqlite_path": file_path,
            "output_dir": output_dir / twoD_type / f"{well_fov}",
        }


# In[10]:


output_dict_of_dfs = {}
for sqlite_dir in [max_projected_input, middle_slice_input, middle_n_slice_input]:
    output_dict_of_dfs[sqlite_dir.name.split("_out")[0].split("cellprofiler_")[1]] = {
        "df_list": [],
    }
output_dict_of_dfs


# In[11]:


total = 0
errors = 0
for featurization_type in tqdm.tqdm(
    well_fov_dict.keys(), leave=True, desc="Processing featurization types"
):
    for well_fov, file_info in tqdm.tqdm(
        well_fov_dict[featurization_type].items(),
        leave=False,
        desc="Processing well-fov pairs",
    ):
        well = well_fov.split("-")[0]
        fov = well_fov.split("-")[1]
        sqlite_file = file_info["sqlite_path"]
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
            output_dict_of_dfs[featurization_type]["df_list"].append(
                full_schema_for_organoid
            )
            continue


# In[12]:


# read in the dataframes and concatenate them in place
for featurization_type in tqdm.tqdm(
    output_dict_of_dfs.keys(), leave=True, desc="Concatenating dataframes"
):
    output_dict_of_dfs[featurization_type]["df"] = pd.concat(
        output_dict_of_dfs[featurization_type]["df_list"], ignore_index=True
    )

    # If any, drop rows where "Metadata_ImageNumber" is NaN (artifact of cytotable)
    output_dict_of_dfs[featurization_type]["df"] = output_dict_of_dfs[
        featurization_type
    ]["df"].dropna(subset=["Metadata_ImageNumber"])
    if featurization_type == "zmax_proj":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            max_projected_organoid_output, index=False
        )
    elif featurization_type == "middle_slice":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            middle_slice_organoid_output, index=False
        )
    elif featurization_type == "middle_n_slice_max_proj":
        output_dict_of_dfs[featurization_type]["df"].to_parquet(
            middle_n_slice_organoid_output, index=False
        )

    print(
        f"Saved {featurization_type} data to {output_dict_of_dfs[featurization_type]['df'].shape[0]} rows in {output_dict_of_dfs[featurization_type]['df'].shape[1]} columns"
    )
