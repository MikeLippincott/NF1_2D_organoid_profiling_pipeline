#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the preprocessing environment
conda activate GFF_segmentation

# convert Jupyter notebooks to scripts
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "NF0030" "NF0040" "SARCO219" "SARCO361" )
patient_array=( "NF0030" "NF0040" )
for patient in "${patient_array[@]}"; do
    echo "Processing patient: $patient"

    z_stack_dir="../../data/$patient/zstack_images"

    mapfile -t well_fovs < <(ls -d "$z_stack_dir"/*)

    total_dirs=$(echo "${well_fovs[@]}" | wc -w)
    echo "Total directories: $total_dirs"
    twoD_methods=( "zmax" "middle" )
    # loop through all input directories
    for well_fov in "${well_fovs[@]}"; do
        well_fov=${well_fov%*/}
        well_fov=$(basename "$well_fov")
        echo "Processing well_fov: $well_fov"
        for twoD_method in "${twoD_methods[@]}"; do
            # run Python script for running preprocessing of morphology profiles
            python 0.segment_nuclei.py \
                --patient "$patient" \
                --well_fov "$well_fov" \
                --clip_limit 0.03 \
                --twoD_method "$twoD_method"
            python 1.segment_cells.py \
                --patient "$patient" \
                --well_fov "$well_fov" \
                --clip_limit 0.01 \
                --twoD_method "$twoD_method"
            python 2.segment_organoids.py \
                --patient "$patient" \
                --well_fov "$well_fov" \
                --clip_limit 0.01 \
                --twoD_method "$twoD_method"
        done
    done
done

cd ../ || exit

conda deactivate

echo "Cell segmentation preprocessing completed successfully."
