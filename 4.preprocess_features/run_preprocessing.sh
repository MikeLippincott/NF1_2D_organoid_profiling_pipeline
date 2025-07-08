#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the preprocessing environment
conda activate gff_preprocessing_env

# convert Jupyter notebooks to scripts
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

patient_ids_file="../data/patient_IDs.txt"
# read patient IDs from the file
mapfile -t patient_ids < "$patient_ids_file"

cd scripts/ || exit

for patient in "${patient_ids[@]}"; do
    # remove trailing newline character from patient ID

    # run preprocessing scripts for each patient
    echo "Running preprocessing for patient: $patient"

    # run Python script for running preprocessing of morphology profiles
    python 0.convert_cytotable.py --patient "$patient"
    python 1.single_cell_processing.py --patient "$patient"
done

cd ../ || exit

conda deactivate

echo "IBP completed for all patients."
