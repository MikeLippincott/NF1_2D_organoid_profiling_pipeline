#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the preprocessing environment
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb


input_file="loadfiles/segmentation_loadfile.txt"

git_root=$(git rev-parse --show-toplevel)
UV_RUN=(uv run --project "$git_root" --no-sync)

patient_well_fov_counter=0
# get the number of lines in the input file
total_lines=$(wc -l < "$input_file")

while IFS= read -r line; do
    ((patient_well_fov_counter++))

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"

    patient="${parts[0]}"
    well_fov="${parts[1]}"
    twoD_method="${parts[2]}"

    echo "[$patient_well_fov_counter/$total_lines] $patient - $well_fov - $twoD_method"

    # create the log file
    log_file="./logs/segment_organoids_${patient}_${well_fov}_${twoD_method}.log"
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi
    mkdir -p "$(dirname "$log_file")"
    touch "$log_file"

    {
        "${UV_RUN[@]}" python scripts/0.segment_nuclei.py \
            --patient "$patient" \
            --well_fov "$well_fov" \
            --clip_limit 0.02 \
            --twoD_method "$twoD_method"

        "${UV_RUN[@]}" python scripts/1.segment_cells.py \
            --patient "$patient" \
            --well_fov "$well_fov" \
            --clip_limit 0.04 \
            --twoD_method "$twoD_method"

        "${UV_RUN[@]}" python scripts/2.segment_organoids.py \
            --patient "$patient" \
            --well_fov "$well_fov" \
            --clip_limit 0.04 \
            --twoD_method "$twoD_method"

    } &> "$log_file"

done < "$input_file"
# done < <(tac "$input_file")
#

conda deactivate

echo ""
echo "========================================="
echo "✓ Cell segmentation preprocessing completed successfully!"
