#!/bin/bash
# Delete cached preprocessed data for merged dataset
echo "Deleting cached preprocessed data..."
rm -rf assets/preprocessed_data/potr/merged*
rm -rf assets/preprocessed_data/POTR/merged*
rm -rf assets/preprocessed_data/*merged*
find assets/preprocessed_data -name "*merged*" -exec rm -rf {} + 2>/dev/null

# Also delete experiment outputs that might have cached data
rm -rf experiment_outs/Hypertune/POTR_merged*
rm -rf experiment_outs/Hypertune/potr_merged*

echo "Starting training..."
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled
python run.py --backbone potr --config merged_backright.json --num_folds 10 --hypertune 1 --tune_fresh 1 --ntrials 30
