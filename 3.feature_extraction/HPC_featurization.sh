#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --account=amc-general
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --output="2D_featurization-%j.out"

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    echo "Using Alpine environment"
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_2d_uv_env/.venv"

    # we need to setup and redirect the output of cellprofiler to work with the HPC environment
    # the scratch space on the HPC is not discoverable by the container
    # so we need to make it discoverable by setting the output directory to be in the scratch space and then bind mounting it into the container
    # we only do this prior to using cellprofiler in a container
    export NF_OUTPUT_BASE_DIR="${SLURM_TMPDIR:-${SCRATCH:-$HOME}}/NF1_2D_outputs"
    mkdir -p "$NF_OUTPUT_BASE_DIR"

    # Bind paths into container namespace
    # /gpfs as read-only input, output as read-write
    export APPTAINER_BINDPATH="/gpfs:/gpfs:ro,${NF_OUTPUT_BASE_DIR}:${NF_OUTPUT_BASE_DIR}:rw"

    echo "Using NF_OUTPUT_BASE_DIR=$NF_OUTPUT_BASE_DIR"
    echo "Using APPTAINER_BINDPATH=$APPTAINER_BINDPATH"
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_2d_uv_env/.venv"

    # we need to setup and redirect the output of cellprofiler to work with the HPC environment
    # the scratch space on the HPC is not discoverable by the container
    # so we need to make it discoverable by setting the output directory to be in the scratch space and then bind mounting it into the container
    # we only do this prior to using cellprofiler in a container
    export NF_OUTPUT_BASE_DIR="${SLURM_TMPDIR:-${SCRATCH:-$HOME}}/NF1_2D_outputs"
    mkdir -p "$NF_OUTPUT_BASE_DIR"

    # Bind paths into container namespace
    # /gpfs as read-only input, output as read-write
    export APPTAINER_BINDPATH="/gpfs:/gpfs:ro,${NF_OUTPUT_BASE_DIR}:${NF_OUTPUT_BASE_DIR}:rw"

    echo "Using NF_OUTPUT_BASE_DIR=$NF_OUTPUT_BASE_DIR"
    echo "Using APPTAINER_BINDPATH=$APPTAINER_BINDPATH"
else
    ENV_PATH="$git_root/.venv"
fi

PYTHON_BIN="$ENV_PATH/bin/python3"

$PYTHON_BIN "$git_root/3.feature_extraction/scripts/check_for_file_completion.py"

input_file="$git_root/3.feature_extraction/loadfiles/featurization_loadfile.txt"

cd scripts/ || exit
# set the counter to zero
patient_well_fov_counter=0
# get the number of lines in the input file
total_lines=$(wc -l < "$input_file")


while IFS= read -r line; do
((patient_well_fov_counter++))

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"

    patient="${parts[0]}"
    well_fov="${parts[1]}"

    echo "  [$patient_well_fov_counter/$total_lines] Processing $patient - $well_fov"

    log_file="../logs/featurize_organoids_${patient}_${well_fov}.log"
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi
    mkdir -p "$(dirname "$log_file")"
    touch "$log_file"

    # call cellprofiler to run the analysis
    # this script runs all three max projection methods in parallel
    {
        "$PYTHON_BIN" cp_analysis.py \
            --patient "$patient" \
            --well_fov "$well_fov"
    } &> "$log_file"

# done < <(tac "$input_file")
done < "$input_file"

cd ../ || exit

conda deactivate

echo "Cell segmentation preprocessing completed successfully."
