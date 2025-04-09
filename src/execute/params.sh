#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo")

# List of target domain sets (each entry can be space-separated if multiple domains)
ALL_TGT_DOMAINS=("sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_3")

# Policies to iterate over
POLICIES=(0 1 2)

# Setting IDs to iterate over
SETTING_IDS=(0 1 2 3 4 5 6 7 8 9) 

# Schedules to iterate over
SCHEDULES=("step_0" "oscillating_0" "oscillating_1")

# Array of seeds for reproducibility
seeds=(0 1 2 3 4)