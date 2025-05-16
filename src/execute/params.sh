#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("mnist" "mnist_m" "syn" "svhn")

# List of model names to iterate over
ALL_MODEL_NAMES=("DigitsDGCNN")

# Policies to iterate over
POLICIES=(6)

# Setting IDs to iterate over
SETTING_IDS=(70 71 72 73 74) 

# Image sizes to test different models
IMG_SIZES=(32)

# Schedules to iterate over
SCHEDULES=(
    "quiet_then_low_1" "quiet_then_low_0" 
)
 
# Array of seeds for reproducibility
seeds=(0 1 2)