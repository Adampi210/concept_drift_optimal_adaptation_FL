#!/bin/bash

# List of source domain sets (each entry can be space-separated if multiple domains)
ALL_SRC_DOMAINS=("Books" "Clothing" "Hotel" "Laptop" "Restaurant")

# List of model names to iterate over
ALL_MODEL_NAMES=("TinyBertForSentiment")

# Policies to iterate over
POLICIES=(5)

# Setting IDs to iterate over
SETTING_IDS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29) 

# Schedules to iterate over
SCHEDULES=(
    "burst"
)
 
# Array of seeds for reproducibility
seeds=(0)