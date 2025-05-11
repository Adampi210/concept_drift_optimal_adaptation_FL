#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo" "art_painting" "cartoon" "sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_4")

# Policies to iterate over
POLICIES=(1 2 3)

# Setting IDs to iterate over
SETTING_IDS=(49) 

# Image sizes to test different models
IMG_SIZES=(128)

# Schedules to iterate over
SCHEDULES=(
    "domain_change_burst_0" "domain_change_burst_1"
    "domain_change_burst_2" "domain_change_burst_3"
)
 
# Array of seeds for reproducibility
seeds=(0 1 2 3 4 5 6 7 8 9)