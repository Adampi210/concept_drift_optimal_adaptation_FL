#!/bin/bash

# Source the parameter arrays
source ./params.sh

# Calculate the lengths of each array
len_A=${#ALL_SRC_DOMAINS[@]}
len_B=${#ALL_TGT_DOMAINS[@]}
len_C=${#POLICIES[@]}
len_D=${#SETTING_IDS[@]}
len_E=${#SCHEDULES[@]}
len_F=${#ALL_MODEL_NAMES[@]}  # Add length of model names

# Calculate total combinations
total_combinations=$((len_A * len_B * len_C * len_D * len_E * len_F))

# Submit the SLURM job array
sbatch --array=0-$((total_combinations - 1))%10 run_experiments.sub

echo "Submitted SLURM job with $total_combinations tasks."