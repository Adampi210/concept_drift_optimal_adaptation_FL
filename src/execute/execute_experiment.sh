#!/bin/bash
# run_experiment_multiple_seeds.sh
# This script runs the Python experiment for multiple seeds (0, 1, 2) in parallel on the same GPU.

# ========== PARAMETERS FROM SLURM ==========
MODEL_NAME="$1"     # Model name
SRC_SET="$2"        # Source domains
POLICY_ID="$3"      # Policy ID
SETTING_ID="$4"     # Setting ID
SCHEDULE="$5"       # Schedule type
# SEED "$6"         # Original seed parameter from SLURM. This script now uses fixed seeds 0, 1, 2.
IMG_SIZE="$7"       # Image size

# ========== USER CONFIGURATION ==========
# Path to the Python experiment script
experiment_filename='evaluate_policy.py'

# Directory to store log files
log_dir="../../logs"
mkdir -p "$log_dir"  # Create log directory if it doesn't exist

# Get current timestamp (used as part of the log filename for uniqueness across script runs)
timestamp=$(date +%Y%m%d_%H%M%S)

# Convert spaces in SRC_SET to underscores for a cleaner log filename component
src_log_str=$(echo "$SRC_SET" | tr ' ' '_')

echo "=========================================="
echo "Launching experiments for seeds 0, 1, and 2 in parallel."
echo "Common Parameters:"
echo "  Model Name     : ${MODEL_NAME}"
echo "  Source Domains : ${SRC_SET}"
echo "  Policy ID      : ${POLICY_ID}"
echo "  Setting ID     : ${SETTING_ID}"
echo "  Schedule       : ${SCHEDULE}"
echo "  Image Size     : ${IMG_SIZE}"
echo "=========================================="

# Define the seeds to run
seeds_to_run=(3 4 5 6 7 8 9)

for current_seed in "${seeds_to_run[@]}"; do
    echo # Blank line for readability
    echo "--- Launching experiment for SEED: ${current_seed} ---"

    # Construct a unique log filename including model name, current_seed, image size, and timestamp
    log_file="$log_dir/drift_src_${src_log_str}_model_${MODEL_NAME}_policy_${POLICY_ID}_setting_${SETTING_ID}_schedule_${SCHEDULE}_seed_${current_seed}_imgsize_${IMG_SIZE}_${timestamp}.log"

    # Display the specific configuration for this seed
    echo "  Log file       : ${log_file}"
    echo "----------------------------------------"

    # Execute the Python experiment script in the background
    # Ensure that $SRC_SET is expanded correctly if it contains multiple domains.
    # If --src_domains expects space-separated domains as a single argument, quote "$SRC_SET".
    # If it expects multiple arguments, $SRC_SET (unquoted) is usually correct.
    # The original script used $SRC_SET unquoted, so we maintain that here.
    python3 "$experiment_filename" \
        --model_name "$MODEL_NAME" \
        --seed "$current_seed" \
        --src_domains $SRC_SET \
        --policy_id "$POLICY_ID" \
        --setting_id "$SETTING_ID" \
        --schedule_type "$SCHEDULE" \
        --img_size "$IMG_SIZE" \
        > "$log_file" 2>&1 &
done

echo
echo "All experiments launched. Waiting for completion..."
wait

# The 'wait' command will return the exit status of the last command to exit,
# or the first non-zero exit status if any job failed.
if [ $? -eq 0 ]; then
    echo "All experiments appear to have completed successfully."
else
    echo "One or more experiments may have failed. Please check the log files in '$log_dir' for details."
fi

echo "=========================================="
echo "Script finished."
echo