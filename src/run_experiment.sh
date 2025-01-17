#!/bin/bash

# run_experiment.sh
# This script runs the test_loss_behavior_under_drift.py experiment for multiple setting IDs and seeds with fixed configurations.

# Path to the Python experiment script
experiment_filename='test_loss_behavior_under_drift.py'

# ===========================
# === CONFIGURATION SETUP ===
# ===========================

# Define fixed configurations
# Modify these variables as needed for your experiment
SRC_DOMAINS=("cartoon")             # Source domains (space-separated if multiple)
TGT_DOMAINS=("photo")            # Target domains (space-separated if multiple)
POLICY_ID=3                       # Fixed Policy ID for retraining decisions

# Array of Setting IDs to iterate over
SETTING_IDS=(50 51 52 53 54 55 56 57 58 59)                 # Add or remove Setting IDs as needed

# Array of seeds for reproducibility
seeds=(0 1 2 4 5)                   # Add or remove seeds as needed

# Directory to store log files
log_dir="../logs"
mkdir -p "$log_dir"                # Create log directory if it doesn't exist

# Get current timestamp to append to log filenames for uniqueness
timestamp=$(date +%Y%m%d_%H%M%S)

# =============================
# === EXPERIMENT EXECUTION ===
# =============================

# Outer loop: Iterate over each Setting ID
for SETTING_ID in "${SETTING_IDS[@]}"; do
    # Inner loop: Iterate over each seed
    for seed in "${seeds[@]}"; do
        # Construct a unique log filename based on configuration and seed
        # Replace spaces with underscores for readability
        src_domains_str=$(IFS=, ; echo "${SRC_DOMAINS[*]}")
        tgt_domains_str=$(IFS=, ; echo "${TGT_DOMAINS[*]}")
        log_file="$log_dir/drift_${src_domains_str}_to_${tgt_domains_str}_policy${POLICY_ID}_setting${SETTING_ID}_seed${seed}_${timestamp}.log"

        # Display the current experiment configuration
        echo "=========================================="
        echo "Running experiment with the following configuration:"
        echo "  Source Domains : ${SRC_DOMAINS[*]}"
        echo "  Target Domains : ${TGT_DOMAINS[*]}"
        echo "  Policy ID      : ${POLICY_ID}"
        echo "  Setting ID     : ${SETTING_ID}"
        echo "  Seed           : ${seed}"
        echo "  Logging to     : ${log_file}"
        echo "=========================================="

        # Execute the Python experiment script with the specified arguments
        python3 "$experiment_filename" \
            --seed "$seed" \
            --src_domains "${SRC_DOMAINS[@]}" \
            --tgt_domains "${TGT_DOMAINS[@]}" \
            --policy_id "$POLICY_ID" \
            --setting_id "$SETTING_ID" \
            > "$log_file" 2>&1

        # Check if the Python script executed successfully
        if [ $? -eq 0 ]; then
            echo "Experiment with Setting ID ${SETTING_ID} and Seed ${seed} completed successfully."
        else
            echo "Experiment with Setting ID ${SETTING_ID} and Seed ${seed} failed. Check the log file for details."
        fi

        echo ""  # Add an empty line for readability
    done
done

echo "All experiments have been executed. Check the '$log_dir' directory for log files."
