#!/bin/bash

# Source the parameter arrays
source ./params.sh

# Calculate the lengths of each array
len_A=${#ALL_SRC_DOMAINS[@]}
len_B=${#POLICIES[@]}
len_C=${#SETTING_IDS[@]}
len_D=${#SCHEDULES[@]}
len_E=${#ALL_MODEL_NAMES[@]}
len_F=${#seeds[@]}  # Length of seeds array
len_G=${#IMG_SIZES[@]}  # Length of image sizes array

# Calculate total combinations, including seeds and image sizes
total_combinations=$((len_A * len_B * len_C * len_D * len_E * len_F * len_G))

# Submit the SLURM job array
sbatch --array=0-$((total_combinations - 1))%80 run_experiments.sub

echo "Submitted SLURM job with $total_combinations tasks."