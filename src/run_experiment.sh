#!/bin/bash

# run_experiment.sh
experiment_filename='test_loss_behavior_under_drift.py'

# Array of configurations
configs=(
    "slow_rotation"

)

# Array of seeds
seeds=(0 1 2)

# Create log directory
log_dir="../logs"
mkdir -p $log_dir

# Get current timestamp for log files
timestamp=$(date +%Y%m%d_%H%M%S)

# Loop through all combinations
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        # Create log filename
        log_file="$log_dir/drift_${config}_seed_${seed}_${timestamp}.log"
        
        echo "Running configuration: $config with seed: $seed"
        # Run experiment and log output
        python3 $experiment_filename --seed $seed --config $config > "$log_file" 2>&1 &
        wait
    done
done