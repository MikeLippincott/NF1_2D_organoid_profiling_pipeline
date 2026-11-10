#!/bin/bash

RUN_CYTOTABLE="FALSE"


# convert Jupyter notebooks to scripts
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    echo "Using Alpine environment"
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_2d_uv_env/.venv"
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_2d_uv_env/.venv"
else
    ENV_PATH="$git_root/.venv"
fi

PYTHON_BIN="$ENV_PATH/bin/python3"


cd scripts/ || exit
patient_array_file_path="../../data/patient_IDs.txt"
readarray -t patient_array < "$patient_array_file_path"

for patient in "${patient_array[@]}"; do
    echo "Processing patient: $patient"
    # run Python script for running preprocessing of morphology profiles
    if [ $RUN_CYTOTABLE = "TRUE" ] ; then
        python 0.convert_cytotable.py --patient "$patient"
    fi
    "$PYTHON_BIN" 1.single_cell_processing.py --patient "$patient"
done

"$PYTHON_BIN" 2.combine_patients.py

conda deactivate

cd ../ || exit

echo "Preprocessing of features completed."
