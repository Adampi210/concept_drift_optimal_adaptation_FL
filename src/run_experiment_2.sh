#!/bin/bash

# run_experiment.sh
# This script runs the test_loss_behavior_under_drift.py experiment for multiple
# configurations of source domains, target domains, policies, setting IDs, and seeds.

# ========== USER CONFIGURATION ==========

# Path to the Python experiment script
experiment_filename='test_loss_behavior_under_drift.py'

# List of source domain sets.
# Each entry can have 1 to 3 domains (space-separated).
ALL_SRC_DOMAINS=(
    "photo"
    "photo"
    "photo"
    "cartoon"
    "art_painting"
    "sketch"
    "cartoon"
    "sketch"
)

# List of target domain sets.
# Each entry can have 1 to 3 domains (space-separated).
ALL_TGT_DOMAINS=(
    "sketch"
    "art_painting"
    "cartoon"
    "photo"
    "photo"
    "photo"
    "sketch"
    "cartoon"
)

# Policies to iterate over
POLICIES=(1)

# Setting IDs to iterate over
SETTING_IDS=(0 1 2 3 4 5 6 7 8 9 10 11)

# Array of seeds for reproducibility
seeds=(0 1 2 4 5)

# Directory to store log files
log_dir="../logs"
mkdir -p "$log_dir"  # Create log directory if it doesn't exist

# Get current timestamp to append to log filenames for uniqueness
timestamp=$(date +%Y%m%d_%H%M%S)

# ========== RUN EXPERIMENTS ==========

# Outer loops: for each source domain set, target domain set, policy, setting, and seed
for SRC_SET in "${ALL_SRC_DOMAINS[@]}"; do
    for TGT_SET in "${ALL_TGT_DOMAINS[@]}"; do
        for POLICY_ID in "${POLICIES[@]}"; do
            for SETTING_ID in "${SETTING_IDS[@]}"; do
                for seed in "${seeds[@]}"; do

                    # Convert spaces to underscores for a cleaner log filename
                    src_log_str=$(echo "$SRC_SET" | tr ' ' '_')
                    tgt_log_str=$(echo "$TGT_SET" | tr ' ' '_')

                    # Construct a unique log filename
                    log_file="$log_dir/drift_${src_log_str}_to_${tgt_log_str}_policy${POLICY_ID}_setting${SETTING_ID}_seed${seed}_${timestamp}.log"

                    # Display the current experiment configuration
                    echo "=========================================="
                    echo "Running experiment with the following configuration:"
                    echo "  Source Domains : ${SRC_SET}"
                    echo "  Target Domains : ${TGT_SET}"
                    echo "  Policy ID      : ${POLICY_ID}"
                    echo "  Setting ID     : ${SETTING_ID}"
                    echo "  Seed           : ${seed}"
                    echo "  Log file       : ${log_file}"
                    echo "=========================================="

                    # Execute the Python experiment script
                    # Note that we pass SRC_SET and TGT_SET *as is* (space-separated)
                    python3 "$experiment_filename" \
                        --seed "$seed" \
                        --src_domains $SRC_SET \
                        --tgt_domains $TGT_SET \
                        --policy_id "$POLICY_ID" \
                        --setting_id "$SETTING_ID" \
                        > "$log_file" 2>&1

                    # Check if the Python script executed successfully
                    if [ $? -eq 0 ]; then
                        echo "Experiment completed successfully."
                    else
                        echo "Experiment failed. Check the log file for details."
                    fi

                    echo ""  # Blank line for readability
                done
            done
        done
    done
done

echo "All experiments have been executed. Check the '$log_dir' directory for log files."
