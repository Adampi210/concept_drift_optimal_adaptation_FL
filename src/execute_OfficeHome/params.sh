#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("RealWorld" "Art" "Product" "Clipart")

# List of model names to iterate over
ALL_MODEL_NAMES=("OfficeHomeNet")

# Policies to iterate over
POLICIES=(5)

# Setting IDs to iterate over
SETTING_IDS=(79) 

# Image sizes to test different models
IMG_SIZES=(224)

# Schedules to iterate over
SCHEDULES=(
    "burst" "spikes" "step" "constant" "wave" "decaying_spikes" "seasonal_flux"
)
 
# Array of seeds for reproducibility
seeds=(0)