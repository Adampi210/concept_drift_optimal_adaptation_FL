#!/bin/bash
# train_models.sh

experiment_filename='pretrain_models_cifar10.py'

# Define the seeds to use
declare -a seeds=(0 1 2 3 4)

# Create log directory
log_dir="../logs"
mkdir -p "$log_dir"

# Get current timestamp for log files
timestamp=$(date +%Y%m%d_%H%M%S)

# Loop through all seeds
for seed in "${seeds[@]}"; do
    # Define model and log filenames
    model_filename="model_seed_${seed}.pth"
    log_file="$log_dir/cifar10_seed_${seed}_${timestamp}.log"
    
    echo "Running training with seed: $seed"
    
    # Run training and log output
    python3 "$experiment_filename" \
        --seed "$seed" > "$log_file" 2>&1
    
    echo "Training with seed $seed completed. Logs saved to $log_file"
done

echo "All training runs completed."
