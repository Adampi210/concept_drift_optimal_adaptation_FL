#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo")

# List of target domain sets (each entry can be space-separated if multiple domains)
ALL_TGT_DOMAINS=("sketch")

# Policies to iterate over
POLICIES=(1 2 3 4)

# Setting IDs to iterate over
SETTING_IDS=(0 1 2 3 4)

# Schedules to iterate over
SCHEDULES=("burst_0" "RV_burst_0" "domain_change_burst_0" "domain_change_burst_1" "domain_change_burst_2")

# Array of seeds for reproducibility
seeds=(0 1 2 4 5)