#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("RealWorld" "Art" "Product" "Clipart")

# List of model names to iterate over
ALL_MODEL_NAMES=("OfficeHomeNet")

# Policies to iterate over
POLICIES=(1 2)

# Setting IDs to iterate over
SETTING_IDS=(60) 

# Image sizes to test different models
IMG_SIZES=(224)

# Schedules to iterate over
SCHEDULES=(
    "domain_change_burst_1" "quiet_then_low_1" "step_1" "RV_domain_change_burst_1" 
)
 
# Array of seeds for reproducibility
seeds=(0 1 2)