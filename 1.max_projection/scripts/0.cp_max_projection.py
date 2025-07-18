#!/usr/bin/env python
# coding: utf-8

# # Perform maximum projection and save the images

# ## Import libraries

# In[ ]:


import argparse
import pathlib

import tifffile
import tqdm

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(in_notebook)


# In[ ]:


if not in_notebook:
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--patient",
        type=str,
        help="Patient ID to use for the segmentation",
    )

    args = parser.parse_args()
    patient = args.patient

else:
    patient = "NF0014"


# In[ ]:


# input images directory
images_dir = pathlib.Path(f"../../data/{patient}/zstack_images/").resolve(strict=True)
# output images directory
output_dir = pathlib.Path(f"../../data/{patient}/zmax_proj/").resolve()


# In[3]:


# get a list of all of the tiff files in the directory
tiff_files = list(images_dir.rglob("*.tif"))
tiff_files.sort()
for tiff_file in tqdm.tqdm(tiff_files):
    # load the first tiff file to get the metadata
    image = tifffile.TiffFile(tiff_file)
    # z max projection
    max_proj = image.asarray().max(axis=0)
    # save the middle slice as a new tiff file
    output_file_dir = output_dir / str(tiff_file.parent).split("/")[-1] / tiff_file.name
    output_file_dir.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_file_dir, max_proj)
