#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("photo" "art_painting" "cartoon" "sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("PACSCNN_4")

# Policies to iterate over
POLICIES=(6)

# Setting IDs to iterate over
SETTING_IDS=(60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79) 

# Image sizes to test different models
IMG_SIZES=(128)

# Schedules to iterate over
SCHEDULES=(
    "domain_change_burst_2" "domain_change_burst_3" 
    "constant_drift_domain_change_0" "constant_drift_domain_change_1" "constant_drift_domain_change_2"
    "sine_wave_domain_change_0" "sine_wave_domain_change_1" "sine_wave_domain_change_2"
    "step_0" "step_1" "step_2"
)
 
# Array of seeds for reproducibility
seeds=(0 1 2 3 4 5 6 7 8 9)