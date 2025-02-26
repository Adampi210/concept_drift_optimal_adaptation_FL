#!/bin/bash

# Source the parameter arrays
source ./params.sh

# Calculate the lengths of each array
len_A=${#ALL_SRC_DOMAINS[@]}
len_B=${#ALL_TGT_DOMAINS[@]}
len_C=${#POLICIES[@]}
len_D=${#SETTING_IDS[@]}
len_E=${#SCHEDULES[@]}

# Calculate total combinations (excluding seeds, which run sequentially within each task)
total_combinations=$((len_A * len_B * len_C * len_D * len_E))

# Submit the SLURM job with the array range 0 to (total_combinations - 1)
sbatch --array=0-$((total_combinations - 1)) run_experiments.sub

echo "Submitted SLURM job with $total_combinations tasks."