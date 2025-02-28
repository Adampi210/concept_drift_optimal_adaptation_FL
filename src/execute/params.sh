#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo" "sketch" "cartoon" "art_painting" "photo")

# List of target domain sets (each entry can be space-separated if multiple domains)
ALL_TGT_DOMAINS=("sketch" "cartoon" "art_painting" "photo")

# Policies to iterate over
POLICIES=(0 1 2 3)

# Setting IDs to iterate over
SETTING_IDS=(0 1 2 3 4 5 6)

# Schedules to iterate over
SCHEDULES=("burst_0" "burst_1" "RV_burst_0" "RV_burst_1" "domain_change_burst_0" "domain_change_burst_1" "domain_change_burst_2" "oscillating_0" "oscillating_1" "step_0" "oscillating_increasing_0")

# Array of seeds for reproducibility
seeds=(0 1 2 4 5 6 7 8 9 10)