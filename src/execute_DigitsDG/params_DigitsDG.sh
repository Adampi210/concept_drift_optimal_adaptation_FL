#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("mnist" "mnist_m" "svhn")

# List of model names to iterate over
ALL_MODEL_NAMES=("DigitsDGCNN")

# Policies to iterate over
POLICIES=(6)

# Setting IDs to iterate over
SETTING_IDS=(40) 

# Image sizes to test different models
IMG_SIZES=(32)

# Schedules to iterate over
SCHEDULES=(
    "step_1"
)
 
# Array of seeds for reproducibility
seeds=(0)