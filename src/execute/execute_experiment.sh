#!/bin/bash
# run_experiment.sh
# This script runs the Python experiment for a single combination of parameters, seed, and image size.

# ========== PARAMETERS FROM SLURM ==========
MODEL_NAME="$1"     # Model name
SRC_SET="$2"        # Source domains
POLICY_ID="$3"      # Policy ID
SETTING_ID="$4"     # Setting ID
SCHEDULE="$5"       # Schedule type
SEED="$6"           # Seed
IMG_SIZE="$7"       # Image size

# ========== USER CONFIGURATION ==========
# Path to the Python experiment script
experiment_filename='evaluate_policy.py'

# Directory to store log files
log_dir="../../logs"
mkdir -p "$log_dir"  # Create log directory if it doesn't exist

# Get current timestamp for log filenames
timestamp=$(date +%Y%m%d_%H%M%S)

# ========== RUN EXPERIMENT ==========
# Convert spaces to underscores for a cleaner log filename
src_log_str=$(echo "$SRC_SET" | tr ' ' '_')

# Construct a unique log filename including model name, seed, and image size
log_file="$log_dir/drift_src_${src_log_str}_model_${MODEL_NAME}_policy_${POLICY_ID}_setting_${SETTING_ID}_schedule_${SCHEDULE}_seed_${SEED}_imgsize_${IMG_SIZE}_${timestamp}.log"

# Display the current experiment configuration
echo "=========================================="
echo "Running experiment with the following configuration:"
echo "  Model Name     : ${MODEL_NAME}"
echo "  Source Domains : ${SRC_SET}"
echo "  Policy ID      : ${POLICY_ID}"
echo "  Setting ID     : ${SETTING_ID}"
echo "  Schedule       : ${SCHEDULE}"
echo "  Seed           : ${SEED}"
echo "  Image Size     : ${IMG_SIZE}"
echo "  Log file       : ${log_file}"
echo "=========================================="

# Execute the Python experiment script
python3 "$experiment_filename" \
    --model_name "$MODEL_NAME" \
    --seed "$SEED" \
    --src_domains $SRC_SET \
    --policy_id "$POLICY_ID" \
    --setting_id "$SETTING_ID" \
    --schedule_type "$SCHEDULE" \
    --img_size "$IMG_SIZE" \
    > "$log_file" 2>&1

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully."
else
    echo "Experiment failed. Check the log file for details."
fi

echo ""  # Blank line for readability