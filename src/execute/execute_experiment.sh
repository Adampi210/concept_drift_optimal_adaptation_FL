#!/bin/bash
# run_experiment.sh
# This script runs the Python experiment for a single combination of parameters, seed, and image size.

# ========== PARAMETERS FROM SLURM ==========
MODEL_NAME="$1"     # Model name
SRC_SET="$2"        # Source domains
TGT_SET="$3"        # Target domains
POLICY_ID="$4"      # Policy ID
SETTING_ID="$5"     # Setting ID
SCHEDULE="$6"       # Schedule type
SEED="$7"           # Seed
IMG_SIZE="$8"       # Image size

# ========== USER CONFIGURATION ==========
# Path to the Python experiment script
experiment_filename='test_loss_behavior_under_drift.py'

# Directory to store log files
log_dir="../../logs"
mkdir -p "$log_dir"  # Create log directory if it doesn't exist

# Get current timestamp for log filenames
timestamp=$(date +%Y%m%d_%H%M%S)

# ========== RUN EXPERIMENT ==========
# Convert spaces to underscores for a cleaner log filename
src_log_str=$(echo "$SRC_SET" | tr ' ' '_')
tgt_log_str=$(echo "$TGT_SET" | tr ' ' '_')

# Construct a unique log filename including model name, seed, and image size
log_file="$log_dir/drift_${src_log_str}_to_${tgt_log_str}_model_${MODEL_NAME}_policy_${POLICY_ID}_setting${SETTING_ID}_schedule${SCHEDULE}_seed${SEED}_imgsize${IMG_SIZE}_${timestamp}.log"

# Display the current experiment configuration
echo "=========================================="
echo "Running experiment with the following configuration:"
echo "  Model Name     : ${MODEL_NAME}"
echo "  Source Domains : ${SRC_SET}"
echo "  Target Domains : ${TGT_SET}"
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
    --tgt_domains $TGT_SET \
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