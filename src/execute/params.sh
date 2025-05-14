#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo" "art_painting" "cartoon" "sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_4")

# Policies to iterate over
POLICIES=(6)

# Setting IDs to iterate over
SETTING_IDS=(60 75 76) 

# Image sizes to test different models
IMG_SIZES=(128)

# Schedules to iterate over
SCHEDULES=(
    "quiet_then_low_0" "RV_domain_change_burst_0" "RV_domain_change_burst_1" "RV_domain_change_burst_2"
)
 
# Array of seeds for reproducibility
seeds=(0 1 2 3 4 5 6 7 8 9)