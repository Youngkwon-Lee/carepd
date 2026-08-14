#!/bin/bash

# LODO (Leave-One-Dataset-Out) Training for POTR
# Train on N-1 datasets, test on 1 dataset
#
# 4 datasets with UPDRS labels (DNE excluded - no labels)

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled

timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/lodo_train

# Config files for 4 datasets with UPDRS labels
declare -A CONFIGS
CONFIGS=(
    ["BMCLab"]="BMCLab.json"
    ["3DGait"]="3DGAIT.json"
    ["PD-GaM"]="PDGAM.json"
    ["T-SDU-PD"]="T-SDU-PD.json"
)

echo "========================================="
echo "LODO Training Started: $timestamp"
echo "========================================="
echo "4 datasets with UPDRS labels"
echo "Each round: Train on 3, Test on 1"
echo "========================================="

for target_dataset in "${!CONFIGS[@]}"; do
    config_file="${CONFIGS[$target_dataset]}"

    echo ""
    echo "🎯 Target (Test): $target_dataset"
    echo "📁 Config: $config_file"
    echo "📚 Training on: all other datasets"

    logfile="./reports/lodo_train/${timestamp}-potr-${target_dataset}_LODO.out"

    python run.py \
        --backbone potr \
        --config "$config_file" \
        --num_folds 6 \
        --hypertune 1 \
        --tune_fresh 1 \
        --ntrials 5 \
        --force_LODO 1 \
        --exp_name_rigid LODO \
        --this_run_num 0 2>&1 | tee "$logfile"

    echo "✅ Completed: $target_dataset"
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo "LODO Training Complete!"
echo "Results saved in: reports/lodo_train/"
echo "========================================="
