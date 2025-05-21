#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("mnist" "mnist_m" "syn" "svhn")

# List of model names to iterate over
ALL_MODEL_NAMES=("DigitsDGCNN")

# Policies to iterate over
POLICIES=(1 2 3 4)

# Setting IDs to iterate over
SETTING_IDS=(60) 

# Image sizes to test different models
IMG_SIZES=(32)

# Schedules to iterate over
SCHEDULES=(
    "constant_drift_domain_change_2" "sine_wave_domain_change_0" "decaying_spikes" "seasonal_flux"
)
 
# Array of seeds for reproducibility
seeds=(0)