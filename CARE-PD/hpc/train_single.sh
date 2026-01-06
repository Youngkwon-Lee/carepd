#!/bin/bash
# CARE-PD Single Experiment Script
# Usage: bash train_single.sh <backbone> <dataset> [num_folds]
# Example: bash train_single.sh motionbert BMCLab 6

BACKBONE=${1:-"motionbert"}
DATASET=${2:-"BMCLab"}
NUM_FOLDS=${3:-6}

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate carepd

cd ~/CARE-PD

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="out/${BACKBONE}_${DATASET}_${NUM_FOLDS}fold_${TIMESTAMP}.log"

echo "=========================================="
echo "CARE-PD Training"
echo "=========================================="
echo "Backbone: $BACKBONE"
echo "Dataset: $DATASET"
echo "Folds: $NUM_FOLDS"
echo "Log: $LOG_FILE"
echo "Started: $(date)"
echo "=========================================="

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run training
python run.py \
    --backbone $BACKBONE \
    --dataset $DATASET \
    --num_folds $NUM_FOLDS \
    --epochs 100 \
    2>&1 | tee $LOG_FILE

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Finished: $(date)"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
