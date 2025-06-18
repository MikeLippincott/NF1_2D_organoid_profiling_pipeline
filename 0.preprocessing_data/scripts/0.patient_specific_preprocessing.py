#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib
import sys

# ## Set paths and variables

# In[2]:


argparse = argparse.ArgumentParser(
    description="Copy files from one directory to another"
)
argparse.add_argument("--HPC", action="store_true", help="Type of compute to run on")

# Parse arguments
args = argparse.parse_args(args=sys.argv[1:] if "ipykernel" not in sys.argv[0] else [])
HPC = args.HPC

print(f"HPC: {HPC}")


# In[3]:


if HPC:
    raw_image_dir_hpc = pathlib.Path("/pl/active/koala/GFF_Data/GFF-Raw/").resolve(
        strict=True
    )
else:
    # comment out depending on whose computer you are on
    # mike's computer
    raw_image_dir_local = pathlib.Path(
        "/home/lippincm/Desktop/18TB/NF1_Patient_organoids/"
    ).resolve(strict=True)
    # Jenna's computer
    # raw_image_dir_local = pathlib.Path("/media/18tbdrive/GFF_organoid_data/Cell Painting-NF0014 Thawed3-Pilot Drug Screening")


# ## NF0014 specific preprocessing

# In[4]:


# Define parent and destination directories in a single dictionary
parent_dir_NF0014_parent = pathlib.Path(
    f"{raw_image_dir_local}/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
    if not HPC
    else f"{raw_image_dir_hpc}/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
).resolve(strict=True)
# get all dirs in the parent dir
parent_dir_NF0014 = list(parent_dir_NF0014_parent.glob("*/"))
parent_dir_NF0014 = [x for x in parent_dir_NF0014 if x.is_dir()]


# In[5]:


for parent_dir in parent_dir_NF0014:
    new_parent_dir = pathlib.Path(
        f"{parent_dir.parent}/{str(parent_dir.stem).split(' (')[0]}"
    )
    # rename the parent dir
    parent_dir.rename(new_parent_dir)
