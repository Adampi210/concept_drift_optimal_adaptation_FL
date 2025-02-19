#!/bin/bash

# Path to the Python experiment script
experiment_filename='test_loss_behavior_under_drift_cifar10.py'

# All drift types to test
DRIFT_TYPES=(
    "color_wash"
)

# Policy to test (Lyapunov optimization policy)
POLICY_ID=2

# Setting IDs to test
SETTING_IDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

# Seeds for reproducibility
SEEDS=(0 1 2 3 4 5)

# Directory to store log files
log_dir="logs"
mkdir -p "$log_dir"

# Get current timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Run experiments
for drift_type in "${DRIFT_TYPES[@]}"; do
    for setting_id in "${SETTING_IDS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Construct log filename
            log_file="$log_dir/drift_${drift_type}_policy${POLICY_ID}_setting${setting_id}_seed${seed}_${timestamp}.log"
            
            # Display current configuration
            echo "=========================================="
            echo "Running experiment with configuration:"
            echo "  Drift Type     : ${drift_type}"
            echo "  Policy ID      : ${POLICY_ID}"
            echo "  Setting ID     : ${setting_id}"
            echo "  Seed           : ${seed}"
            echo "  Log file       : ${log_file}"
            echo "=========================================="
            
            # Execute Python script
            python "$experiment_filename" \
                --seed "$seed" \
                --drift_type "$drift_type" \
                --policy_id "$POLICY_ID" \
                --setting_id "$setting_id" \
                --n_rounds 200 \
                > "$log_file" 2>&1
            
            # Check execution status
            if [ $? -eq 0 ]; then
                echo "✓ Experiment completed successfully"
            else
                echo "✗ Experiment failed. Check ${log_file} for details"
            fi
            echo ""
        done
    done
done

echo "All experiments completed. Log files are in '${log_dir}' directory."