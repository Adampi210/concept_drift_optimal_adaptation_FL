#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("mnist" "mnist_m" "syn" "svhn")

# List of model names to iterate over
ALL_MODEL_NAMES=("DigitsDGCNN")

# Policies to iterate over
POLICIES=(1 2 3 4)

# Setting IDs to iterate over
SETTING_IDS=(40 41 42 43 44 45 46 47 48 49) 

# Image sizes to test different models
IMG_SIZES=(32)

# Schedules to iterate over
SCHEDULES=(
    "domain_change_burst_1" "quiet_then_low_1" "step_1" "RV_domain_change_burst_1" 
)
 
# Array of seeds for reproducibility
seeds=(0 1 2)