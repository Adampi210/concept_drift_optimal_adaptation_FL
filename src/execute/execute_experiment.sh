#!/bin/bash
# run_experiment.sh
# This script runs the Python experiment for a single combination of parameters,
# looping over seeds sequentially.

# ========== PARAMETERS FROM SLURM ==========

SRC_SET="$1"        # Source domains
TGT_SET="$2"        # Target domains
POLICY_ID="$3"      # Policy ID
SETTING_ID="$4"     # Setting ID
SCHEDULE="$5"       # Schedule type

# ========== USER CONFIGURATION ==========

# Path to the Python experiment script
experiment_filename='test_loss_behavior_under_drift.py'

# Array of seeds for reproducibility
seeds=(0 1 2 4 5)

# Directory to store log files
log_dir="../logs"
mkdir -p "$log_dir"  # Create log directory if it doesn't exist

# Get current timestamp for log filenames
timestamp=$(date +%Y%m%d_%H%M%S)

# ========== RUN EXPERIMENTS ==========

# Loop over seeds for this combination
for seed in "${seeds[@]}"; do
    # Convert spaces to underscores for a cleaner log filename
    src_log_str=$(echo "$SRC_SET" | tr ' ' '_')
    tgt_log_str=$(echo "$TGT_SET" | tr ' ' '_')

    # Construct a unique log filename
    log_file="$log_dir/drift_${src_log_str}_to_${tgt_log_str}_policy${POLICY_ID}_setting${SETTING_ID}_schedule${SCHEDULE}_seed${seed}_${timestamp}.log"

    # Display the current experiment configuration
    echo "=========================================="
    echo "Running experiment with the following configuration:"
    echo "  Source Domains : ${SRC_SET}"
    echo "  Target Domains : ${TGT_SET}"
    echo "  Policy ID      : ${POLICY_ID}"
    echo "  Setting ID     : ${SETTING_ID}"
    echo "  Schedule       : ${SCHEDULE}"
    echo "  Seed           : ${seed}"
    echo "  Log file       : ${log_file}"
    echo "=========================================="

    # Execute the Python experiment script
    python3 "$experiment_filename" \
        --seed "$seed" \
        --src_domains $SRC_SET \
        --tgt_domains $TGT_SET \
        --policy_id "$POLICY_ID" \
        --setting_id "$SETTING_ID" \
        --schedule_type "$SCHEDULE" \
        > "$log_file" 2>&1

    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Experiment completed successfully."
    else
        echo "Experiment failed. Check the log file for details."
    fi

    echo ""  # Blank line for readability
done

echo "All seeds for this combination have been executed."