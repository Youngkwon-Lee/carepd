#!/usr/bin/env python3
"""
Merge all 9 CARE-PD datasets into a unified dataset.
Creates combined npz files for training on the full dataset.

Usage:
    python scripts/merge_datasets.py --view backright
    python scripts/merge_datasets.py --view sideright
"""

import numpy as np
import os
import argparse
from pathlib import Path
from collections import defaultdict

# Dataset configurations
DATASETS = {
    # Regular datasets (no suffix)
    '3DGait': {'suffix': '', 'has_labels': True},
    'BMCLab': {'suffix': '', 'has_labels': True},
    'DNE': {'suffix': '', 'has_labels': True},
    'E-LC': {'suffix': '', 'has_labels': True},
    'KUL-DT-T': {'suffix': '', 'has_labels': True},
    'PD-GaM': {'suffix': '', 'has_labels': True},
    # Slope corrected datasets
    'T-LTC': {'suffix': '_slopeCorrected', 'has_labels': True},
    'T-SDU': {'suffix': '_slopeCorrected', 'has_labels': False},  # No UPDRS labels
    'T-SDU-PD': {'suffix': '_slopeCorrected', 'has_labels': True},
}

def get_npz_path(base_dir, dataset, view, suffix):
    """Get the path to the npz file for a dataset."""
    filename = f'h36m_3d_world2cam_{view}_floorXZZplus_30f_or_longer{suffix}.npz'
    return base_dir / dataset / filename

def load_dataset(npz_path, dataset_name):
    """Load a single dataset and return sequences with metadata."""
    if not npz_path.exists():
        print(f"  WARNING: {npz_path} not found, skipping...")
        return None

    data = np.load(npz_path, allow_pickle=True)

    # Get all keys (sequence names)
    sequences = {}
    labels = {}
    subjects = set()

    for key in data.files:
        seq_data = data[key]
        if isinstance(seq_data, np.ndarray) and seq_data.ndim == 0:
            seq_data = seq_data.item()

        # Create unique key with dataset prefix
        unique_key = f"{dataset_name}_{key}"
        sequences[unique_key] = seq_data

        # Extract subject ID from key (format: SUBxx_walkxx or similar)
        parts = key.split('_')
        if len(parts) >= 1:
            subject_id = f"{dataset_name}_{parts[0]}"
            subjects.add(subject_id)

    print(f"  Loaded {len(sequences)} sequences from {dataset_name} ({len(subjects)} subjects)")
    return {
        'sequences': sequences,
        'subjects': subjects,
        'dataset': dataset_name
    }

def merge_datasets(base_dir, view, output_dir):
    """Merge all datasets into a single npz file."""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sequences = {}
    all_subjects = set()
    dataset_stats = {}

    print(f"\n{'='*60}")
    print(f"Merging datasets for view: {view}")
    print(f"{'='*60}\n")

    for dataset_name, config in DATASETS.items():
        print(f"Processing {dataset_name}...")
        npz_path = get_npz_path(base_dir, dataset_name, view, config['suffix'])

        result = load_dataset(npz_path, dataset_name)
        if result is None:
            continue

        all_sequences.update(result['sequences'])
        all_subjects.update(result['subjects'])
        dataset_stats[dataset_name] = {
            'sequences': len(result['sequences']),
            'subjects': len(result['subjects'])
        }

    # Save merged dataset
    output_path = output_dir / f'CARE-PD_merged_{view}.npz'
    print(f"\nSaving merged dataset to {output_path}...")
    np.savez_compressed(output_path, **all_sequences)

    # Save subject list for LOSO fold generation
    subjects_list = sorted(list(all_subjects))
    subjects_path = output_dir / f'CARE-PD_merged_subjects_{view}.txt'
    with open(subjects_path, 'w') as f:
        for subj in subjects_list:
            f.write(f"{subj}\n")

    # Print summary
    print(f"\n{'='*60}")
    print("MERGE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTotal sequences: {len(all_sequences)}")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"\nPer-dataset breakdown:")
    print(f"{'Dataset':<15} {'Sequences':>12} {'Subjects':>10}")
    print("-" * 40)
    for ds, stats in sorted(dataset_stats.items()):
        print(f"{ds:<15} {stats['sequences']:>12} {stats['subjects']:>10}")
    print("-" * 40)
    print(f"{'TOTAL':<15} {len(all_sequences):>12} {len(all_subjects):>10}")

    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {subjects_path}")

    return {
        'total_sequences': len(all_sequences),
        'total_subjects': len(all_subjects),
        'dataset_stats': dataset_stats,
        'output_path': str(output_path)
    }

def main():
    parser = argparse.ArgumentParser(description='Merge CARE-PD datasets')
    parser.add_argument('--view', type=str, default='backright',
                        choices=['backright', 'sideright'],
                        help='Camera view to merge')
    parser.add_argument('--base_dir', type=str,
                        default='./assets/datasets/h36m',
                        help='Base directory containing dataset folders')
    parser.add_argument('--output_dir', type=str,
                        default='./assets/datasets/h36m/merged',
                        help='Output directory for merged dataset')

    args = parser.parse_args()

    result = merge_datasets(args.base_dir, args.view, args.output_dir)

    print(f"\n✅ Successfully merged {result['total_sequences']} sequences from {result['total_subjects']} subjects!")

if __name__ == '__main__':
    main()
