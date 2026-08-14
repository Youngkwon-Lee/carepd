#!/bin/bash

# LODO Evaluation - Generate Predictions
# Generate predictions from best models for each LODO dataset

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled

timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/lodo_predictions

# 4 datasets with UPDRS labels
declare -A CONFIGS
CONFIGS=(
    ["BMCLab"]="BMCLab.json"
    ["3DGait"]="3DGAIT.json"
    ["PD-GaM"]="PDGAM.json"
    ["T-SDU-PD"]="T-SDU-PD.json"
)

# Best trials from hypertuning
declare -A BEST_TRIALS
BEST_TRIALS=(
    ["BMCLab"]="0"
    ["3DGait"]="4"
    ["PD-GaM"]="2"
    ["T-SDU-PD"]="1"
)

echo "========================================="
echo "LODO Prediction Generation Started: $timestamp"
echo "========================================="

for target_dataset in "${!CONFIGS[@]}"; do
    config_file="${CONFIGS[$target_dataset]}"
    best_trial="${BEST_TRIALS[$target_dataset]}"

    echo ""
    echo "🎯 Target Dataset: $target_dataset"
    echo "📁 Config: $config_file"
    echo "🏆 Best Trial: $best_trial"

    logfile="./reports/lodo_predictions/${timestamp}-potr-${target_dataset}_predictions.out"

    # Evaluate on test set using best trial model
    python run.py \
        --backbone potr \
        --config "$config_file" \
        --num_folds 6 \
        --hypertune 0 \
        --force_LODO 1 \
        --exp_name_rigid "LODO_predictions" \
        --this_run_num "$best_trial" \
        --readstudyfrom "$best_trial" 2>&1 | tee "$logfile"

    echo "✅ Completed: $target_dataset"
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo "LODO Predictions Generated!"
echo "Results saved in: experiment_outs/LODO/*/"
echo "========================================="
