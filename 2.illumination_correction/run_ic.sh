#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the CellProfiler environment
conda activate gff_cp_env

# convert Jupyter notebooks to scripts
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb


cd scripts/ || exit 1

# patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "NF0030" "NF0040" "SARCO219" "SARCO361" )
patient_array=( "NF0030" "NF0040" )

for patient in "${patient_array[@]}"; do
echo "Processing patient: $patient"
    # run Python script for performing illumination correction with CellProfiler
    python cp_illum_correction.py --patient "$patient"
done

conda deactivate

cd ../ || exit 1

echo "Illumination correction completed successfully."
