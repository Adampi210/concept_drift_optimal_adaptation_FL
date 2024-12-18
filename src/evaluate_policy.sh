#!/bin/bash

# evaluate_policy.sh
experiment_filename='test_loss_behavior_under_drift.py'

# List of available domains
domains=("photo" "cartoon" "sketch" "art_painting")

# List of seeds for reproducibility
seeds=(0)

# Policy and hyperparameters (fixed for now)
policy_id=0
setting_id=0
alpha=1.0
beta=1.0
drift_rate=0.1  # Adjust as needed
n_rounds=100    # Adjust as needed
learning_rate=0.001  # Adjust as needed

# Directories for logs and models
eval_log_dir="../logs"
model_save_dir="../../../models/concept_drift_models"
mkdir -p "$eval_log_dir"
mkdir -p "$model_save_dir"

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

# Function to generate all combinations of 1 to N elements
generate_all_combinations() {
    local max_size=$1
    shift
    local elements=("$@")
    local size
    local combos=()
    for size in $(seq 1 $max_size); do
        while read -r combo; do
            combos+=("$combo")
        done < <(generate_combinations $size "${elements[@]}")
    done
    echo "${combos[@]}"
}

# Generate all source and target domain combinations
# Source domains: combinations of 1 to 3 domains
source_combinations=($(generate_all_combinations 3 "${domains[@]}"))

# Target domains: combinations of 1 to 4 domains
target_combinations=($(generate_all_combinations 4 "${domains[@]}"))

# Loop through all source, target, and seed combinations
for src_combo in "${source_combinations[@]}"; do
    for tgt_combo in "${target_combinations[@]}"; do
        for seed in "${seeds[@]}"; do
            
            # Replace spaces with underscores for filenames
            src_str=$(echo "$src_combo" | tr ' ' '_')
            tgt_str=$(echo "$tgt_combo" | tr ' ' '_')
            
            # Define model filename (Assuming models are saved with this naming convention)
            model_filename="model_domains_${src_str}.pth"
            model_path="$model_save_dir/$model_filename"
            
            # Check if the model file exists
            if [ ! -f "$model_path" ]; then
                echo "Model file $model_path does not exist. Skipping evaluation for Source: [$src_combo], Target: [$tgt_combo], Seed: $seed."
                continue
            fi
            
            # Define evaluation log file
            eval_log_file="$eval_log_dir/eval_src_${src_str}_tgt_${tgt_str}_seed_${seed}_${timestamp}.log"
            
            echo "========================================"
            echo "Evaluating: Source Domains = [$src_combo], Target Domains = [$tgt_combo], Seed = $seed"
            
            # Run evaluation and log output
            python3 "$experiment_filename" evaluate \
                --seed "$seed" \
                --src_domains $src_combo \
                --tgt_domains $tgt_combo \
                --drift_rate "$drift_rate" \
                --n_rounds "$n_rounds" \
                --lr "$learning_rate" \
                --policy_id "$policy_id" \
                --setting_id "$setting_id" \
                --alpha "$alpha" \
                --beta "$beta" > "$eval_log_file" 2>&1
            
            if [ $? -eq 0 ]; then
                echo "Evaluation completed for Source: [$src_combo], Target: [$tgt_combo], Seed: $seed. Logs saved to $eval_log_file"
            else
                echo "Evaluation FAILED for Source: [$src_combo], Target: [$tgt_combo], Seed: $seed. Check $eval_log_file for details."
            fi
            
            echo "========================================"
            
        done
    done
done

echo "All evaluation runs completed."
