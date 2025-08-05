#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("real" "painting" "clipart" "sketch")

# List of model names to iterate over
ALL_MODEL_NAMES=("DomainNetNet")

# Policies to iterate over
POLICIES=(5)

# Setting IDs to iterate over
SETTING_IDS=(71 72 73 74) 

# Image sizes to test different models
IMG_SIZES=(224)

# Schedules to iterate over
SCHEDULES=(
    "burst"
)
 
# Array of seeds for reproducibility
seeds=(0)