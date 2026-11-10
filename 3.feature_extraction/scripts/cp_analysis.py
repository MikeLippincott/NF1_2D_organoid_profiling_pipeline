#!/usr/bin/env python
# coding: utf-8

# # Perform analysis (segmentation and feature extraction) and save features as an SQLite database

# ## Import libraries

# In[1]:


import os
import pathlib
import pprint

import psutil
from image_analysis_2D.cp_utils.cp_utils import run_cellprofiler
from image_analysis_2D.featurization_utils.resource_profiling_utils import (
    start_resource_profiling,
    stop_resource_profiling,
)
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
    well_fov = args_dict["well_fov"]
    check_for_missing_args(
        patient=patient,
        well_fov=well_fov,
    )

else:
    print("Running in a notebook")
    well_fov = "G9-2"
    patient = "NF0037_T1"


max_projected_input = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/0a.zmax_proj/{well_fov}"
).resolve(strict=True)
middle_slice_input = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/0b.middle_slice/{well_fov}"
).resolve(strict=True)
middle_n_input = pathlib.Path(
    f"{image_base_dir}/data/{patient}/2D_analysis/0c.middle_n_slice_max_proj/{well_fov}"
).resolve(strict=True)


# ## Set paths and variables

# In[3]:


# set the run type for the parallelization
run_name = f"{patient}_{well_fov}"

# set path for CellProfiler pipeline
path_to_pipeline_sc = pathlib.Path(
    f"{root_dir}/3.feature_extraction/pipelines/analysis_single_cell.cppipe"
).resolve(strict=True)
path_to_pipeline_organoid = pathlib.Path(
    f"{root_dir}/3.feature_extraction/pipelines/analysis_organoid.cppipe"
).resolve(strict=True)
# Get the plate name from the folder name
plate_name = f"{patient}_{well_fov}"  # Get the folder name as the plate name

output_base_dir = f"{root_dir}"


# ## Create dictionary to process data

# In[4]:


plate_info_dictionary = {}
# create plate info dictionary with all parts of the CellProfiler CLI command to run in parallel
for images_dir in [middle_slice_input, max_projected_input, middle_n_input]:
    if "zmax_proj" in str(images_dir):
        output_path = f"{output_base_dir}/data/{patient}/2D_analysis/2a.cellprofiler_{str(images_dir.parent.name.split('0a.')[1])}_output/{well_fov}/"
    elif "middle_slice" in str(images_dir):
        output_path = f"{output_base_dir}/data/{patient}/2D_analysis/2b.cellprofiler_{str(images_dir.parent.name.split('0b.')[1])}_output/{well_fov}/"
    elif "middle_n" in str(images_dir):
        output_path = f"{output_base_dir}/data/{patient}/2D_analysis/2c.cellprofiler_{str(images_dir.parent.name.split('0c.')[1])}_output/{well_fov}/"
    for object_type in ["single_cell", "organoid"]:
        pipeline = (
            path_to_pipeline_sc
            if object_type == "single_cell"
            else path_to_pipeline_organoid
        )

        plate_info_dictionary[
            f"{plate_name}_{str(images_dir.parent.name)}_{object_type}"
        ] = {
            "path_to_images": images_dir,
            "path_to_output": pathlib.Path(output_path).resolve(),
            "path_to_pipeline": pipeline,
        }

# view the dictionary to assess that all info is added correctly
if in_notebook:
    pprint.pprint(plate_info_dictionary, indent=4)


# In[5]:


# check if there is a sqlite db already present, if so remove the run from the dictionary
plates_to_run = {}
for plate_name, plate_info in plate_info_dictionary.items():
    if not plate_info["path_to_output"].exists():
        plate_info["path_to_output"].mkdir(parents=True, exist_ok=True)
    sqlite_files = list(plate_info["path_to_output"].glob("*.sqlite"))
    if len(sqlite_files) == 0:
        plates_to_run[plate_name] = plate_info
    else:
        print(
            f"SQLite database already present for {plate_name}, skipping CellProfiler run."
        )
if in_notebook:
    pprint.pprint(plates_to_run, indent=4)


# ## Perform CellProfiler analysis on data
#
# The function being called is called "run_cellprofiler_parallel" but can be used if there is only one plate to run. We can also split the data by well and process that way in parallel, but we choose to process at all at once for now.
#
# Note: This code cell was not ran as we prefer to perform CellProfiler processing tasks via `sh` file (bash script) which is more stable.

# In[6]:


try:
    path_to_apptainer_image = pathlib.Path(
        f"{root_dir}/environments/cellprofiler.sif"
    ).resolve(strict=True)
    print("Using apptainer image for CellProfiler run.")
except FileNotFoundError:
    print("No apptainer image found, running CellProfiler without apptainer.")
    path_to_apptainer_image = None


# In[7]:


if len(plates_to_run) == 0:
    print("All runs have already ran")
else:
    for plate_name, plate_info in tqdm.tqdm(plates_to_run.items()):
        start_time, start_memory = start_resource_profiling()
        run_cellprofiler(
            path_to_pipeline=plate_info["path_to_pipeline"],
            path_to_input=plate_info["path_to_images"],
            path_to_output=plate_info["path_to_output"],
            run_with_apptainer_interactive=path_to_apptainer_image,
            log_file_name=f"{plate_name}.log",
        )
        # move the log file to the output directory
        stop_resource_profiling(
            start_time=start_time,
            well_fov=well_fov,
            patient_id=patient,
            feature_type="cellprofiler2D",
            channel="All",
            compartment="All",
            CPU_GPU="CPU",
            output_file_dir=pathlib.Path(
                f"{root_dir}/data/{patient}/2D_analysis/run_stats/{plate_name}_{well_fov}_profiling_stats.parquet"
            ),
            start_mem=start_memory,
        )
