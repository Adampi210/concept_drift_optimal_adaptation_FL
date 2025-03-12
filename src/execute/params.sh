#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo")

# List of target domain sets (each entry can be space-separated if multiple domains)
ALL_TGT_DOMAINS=("sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_1" "PACSCNN_3")

# Policies to iterate over
POLICIES=(4)

# Setting IDs to iterate over
SETTING_IDS=(22 23 24 25 26 27 28 29)

# Schedules to iterate over
SCHEDULES=("domain_change_burst_0")

# Array of seeds for reproducibility
seeds=(0 1 2 3 4)