#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo")

# List of target domain sets (each entry can be space-separated if multiple domains)
ALL_TGT_DOMAINS=("sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_1" "PACSCNN_2" "PACSCNN_3" "PACSCNN_4")

# Policies to iterate over
POLICIES=(0 1 2)

# Setting IDs to iterate over
SETTING_IDS=(5 6 7 8 9) 

# Schedules to iterate over
SCHEDULES=("domain_change_burst_0" "domain_change_burst_1" "domain_change_burst_2")

# Array of seeds for reproducibility
seeds=(0 1 2 3 4)