#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo" "art_painting" "cartoon" "sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_4")

# Policies to iterate over
POLICIES=(6)

# Setting IDs to iterate over
SETTING_IDS=(78 79) 

# Image sizes to test different models
IMG_SIZES=(128)

# Schedules to iterate over
SCHEDULES=(
    "constant_drift_domain_change_0"
)
 
# Array of seeds for reproducibility
seeds=(0)