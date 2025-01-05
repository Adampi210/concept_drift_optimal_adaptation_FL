#!/bin/bash
# train_models.sh

experiment_filename='pretrain_models.py'
# List of available domains
domains=("photo" "cartoon" "sketch" "art_painting")

# Seed for all experiments
seed=3

# Create log directory
log_dir="../logs"
mkdir -p "$log_dir"

# Get current timestamp for log files
timestamp=$(date +%Y%m%d_%H%M%S)

# Define the specific combinations we want
# Format: "domain1 domain2 domain3" (space-separated)
declare -a combinations=(
    "photo"
    "cartoon"
    "sketch"
    "art_painting"
    "photo cartoon"
    "photo sketch"
    "photo art_painting"
    "cartoon sketch"
    "cartoon art_painting"
    "sketch art_painting"
    "photo cartoon sketch"
    "photo cartoon art_painting"
    "photo sketch art_painting"
    "cartoon sketch art_painting"
)


# Loop through all combinations
for combo in "${combinations[@]}"; do
    # Replace spaces with underscores for filenames
    combo_str=$(echo "$combo" | tr ' ' '_')
    
    # Define model and log filenames
    model_filename="model_domains_${combo_str}.pth"
    log_file="$log_dir/drift_domains_${combo_str}_seed_${seed}_${timestamp}.log"
    
    echo "Running training for domains: $combo with seed: $seed"
    
    # Run training and log output
    python3 "$experiment_filename" \
        --seed "$seed" \
        --domains $combo > "$log_file" 2>&1
    
    echo "Training for domains: $combo completed. Logs saved to $log_file"
done

echo "All training runs completed."