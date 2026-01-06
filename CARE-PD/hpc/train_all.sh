#!/bin/bash
# CARE-PD Full Training Script
# Runs all backbone × dataset combinations

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate carepd

cd ~/CARE-PD

# Create output directory
mkdir -p out/hpc_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="out/hpc_results/${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "=========================================="
echo "CARE-PD Full Training"
echo "Started: $(date)"
echo "Log directory: $LOG_DIR"
echo "=========================================="

# Define experiments
BACKBONES=("motionbert" "potr" "momask" "motionagformer")
DATASETS=("BMCLab" "3DGait" "PD-GaM" "T-SDU-PD")

# Dataset-specific fold counts (LOSO)
declare -A FOLDS
FOLDS["BMCLab"]=6      # or 23 for LOSO
FOLDS["3DGait"]=6      # or 43 for LOSO
FOLDS["PD-GaM"]=6      # or 30 for LOSO
FOLDS["T-SDU-PD"]=6    # or 14 for LOSO

# Run experiments
for backbone in "${BACKBONES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        num_folds=${FOLDS[$dataset]}
        log_file="${LOG_DIR}/${backbone}_${dataset}_${num_folds}fold.log"

        echo ""
        echo "[$(date +%H:%M:%S)] Starting: $backbone + $dataset (${num_folds}-fold)"
        echo "  Log: $log_file"

        python run.py \
            --backbone $backbone \
            --dataset $dataset \
            --num_folds $num_folds \
            --epochs 100 \
            > $log_file 2>&1

        # Extract final result
        tail -5 $log_file | head -3
        echo "  Completed: $(date +%H:%M:%S)"
    done
done

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "Finished: $(date)"
echo "Results saved to: $LOG_DIR"
echo "=========================================="

# Summary
echo ""
echo "Results Summary:"
echo "----------------"
for log in ${LOG_DIR}/*.log; do
    name=$(basename $log .log)
    result=$(grep -E "Best F1|Final Accuracy|MAE" $log | tail -1 || echo "No result found")
    echo "$name: $result"
done
