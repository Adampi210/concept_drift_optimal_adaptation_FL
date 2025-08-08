#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo" "art_painting" "cartoon" "sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN")

# Policies to iterate over
POLICIES=(5)

# Setting IDs to iterate over
SETTING_IDS=(70 71 72 73 74 75 76)

# Image sizes to test different models
IMG_SIZES=(128)

# Schedules to iterate over
SCHEDULES=(
    "burst_1" "burst_2"
)
 
# Array of seeds for reproducibility
seeds=(0)