#!/bin/bash

# train_models.sh
experiment_filename='pretrain_models.py'

# List of available domains
domains=("photo" "cartoon" "sketch" "art_painting")

# Seed for all experiments
seed=2

# Create log directory
log_dir="../logs"
mkdir -p "$log_dir"

# Get current timestamp for log files
timestamp=$(date +%Y%m%d_%H%M%S)

# Function to generate combinations
generate_combinations() {
    local n=$1
    shift
    local elements=("$@")
    local i
    if [ "$n" -eq 0 ]; then
        echo ""
        return
    fi
    for ((i=0; i<${#elements[@]}; i++)); do
        local elem=${elements[i]}
        if [ "$n" -eq 1 ]; then
            echo "$elem"
        else
            local rest=("${elements[@]:0:i}" "${elements[@]:i+1}")
            local subcombinations=$(generate_combinations $((n-1)) "${rest[@]}")
            for sub in $subcombinations; do
                echo "$elem $sub"
            done
        fi
    done
}

# Generate all combinations of 1, 2, and 3 domains
combinations=()
for size in 1 2 3; do
    while read -r combo; do
        combinations+=("$combo")
    done < <(generate_combinations $size "${domains[@]}")
done

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
        --domains $combo  > "$log_file" 2>&1
    
    echo "Training for domains: $combo completed. Logs saved to $log_file"
done

echo "All training runs completed."
