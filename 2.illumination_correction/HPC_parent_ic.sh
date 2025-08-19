#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --time=10:00
#SBATCH --qos=normal
#SBATCH --output=illumination_correction-%j.out


# convert Jupyter notebooks to scripts
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

# patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "NF0030" "NF0040" "SARCO219" "SARCO361" )
patient_array=( "NF0030" "NF0040" )
for patient in "${patient_array[@]}"; do
echo "Processing patient: $patient"
    # run Python script for performing illumination correction with CellProfiler
    sbatch HPC_child_ic.sh "$patient"
done


echo "Illumination correction completed successfully."
