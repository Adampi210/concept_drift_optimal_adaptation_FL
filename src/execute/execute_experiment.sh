#!/bin/bash
# run_experiment_multiple_seeds.sh
# This script runs experiments sequentially for each uncertainty type.
# For each uncertainty type, it runs multiple seeds in parallel.

# ========== PARAMETERS FROM SLURM ==========
MODEL_NAME="$1"     # Model name
SRC_SET="$2"        # Source domains
POLICY_ID="$3"      # Policy ID
SETTING_ID="$4"     # Setting ID
SCHEDULE="$5"       # Schedule type
# SEED is now handled by the loop
IMG_SIZE="$7"       # Image size

# ========== USER CONFIGURATION ==========
# Path to the Python experiment script
experiment_filename='evaluate_policy.py'

# Directory to store log files
log_dir="../../logs"
mkdir -p "$log_dir"  # Create log directory if it doesn't exist

# Get current timestamp for a unique run identifier
timestamp=$(date +%Y%m%d_%H%M%S)

# Convert spaces in SRC_SET to underscores for a cleaner log filename component
src_log_str=$(echo "$SRC_SET" | tr ' ' '_')

# --- Define the experimental conditions to run ---
seeds_to_run=(0 1 2)
uncertainty_levels=(
    "none"
    # "gaussian_0"  # 5% proportional Gaussian noise
    # "gaussian_1"  # 10% proportional Gaussian noise
    # "gaussian_2"  # 20% proportional Gaussian noise
    "gaussian_3" # 0.1 fixed bias + 5% proportional noise
    "gaussian_4" # Autocorrelated noise (rho=0.9, 5% scale)
)

echo "=========================================="
echo "Launching experiments."
echo "Common Parameters:"
echo "  Model Name      : ${MODEL_NAME}"
echo "  Source Domains  : ${SRC_SET}"
echo "  Policy ID       : ${POLICY_ID}"
echo "  Setting ID      : ${SETTING_ID}"
echo "  Schedule        : ${SCHEDULE}"
echo "  Image Size      : ${IMG_SIZE}"
echo "=========================================="

# Loop through each uncertainty level sequentially
for uncertainty in "${uncertainty_levels[@]}"; do
    echo # Blank line for readability
    echo "=========================================="
    echo "--- Starting experiments for UNCERTAINTY: ${uncertainty} ---"
    echo "--- Seeds will run in parallel. ---"
    
    # Loop through each seed and launch in parallel for the current uncertainty level
    for current_seed in "${seeds_to_run[@]}"; do
        echo "--- Launching experiment for SEED: ${current_seed} ---"

        # Construct a unique log filename for each specific run
        log_file="$log_dir/drift_src_${src_log_str}_model_${MODEL_NAME}_policy_${POLICY_ID}_setting_${SETTING_ID}_schedule_${SCHEDULE}_seed_${current_seed}_uncertainty_${uncertain}_imgsize_${IMG_SIZE}_${timestamp}.log"
        
        echo "  Log file        : ${log_file}"
        echo "----------------------------------------"

        # Execute the Python experiment script in the background
        # The '&' at the end runs the command as a background job.
        python3 "$experiment_filename" \
            --model_name "$MODEL_NAME" \
            --seed "$current_seed" \
            --src_domains $SRC_SET \
            --policy_id "$POLICY_ID" \
            --setting_id "$SETTING_ID" \
            --schedule_type "$SCHEDULE" \
            --img_size "$IMG_SIZE" \
            --loss_uncertainty "$uncertainty" \
            > "$log_file" 2>&1 &
    done

    echo
    echo "All seeds for uncertainty '${uncertainty}' launched. Waiting for completion..."
    wait # This command waits for all background jobs (for the current uncertainty) to finish.
    
    # Check the exit status of the last command to finish.
    if [ $? -eq 0 ]; then
        echo "All experiments for uncertainty '${uncertainty}' appear to have completed successfully."
    else
        echo "One or more experiments for uncertainty '${uncertainty}' may have failed. Please check the log files."
    fi
    echo "=========================================="

done


echo
echo "All experiment sets finished."
echo
