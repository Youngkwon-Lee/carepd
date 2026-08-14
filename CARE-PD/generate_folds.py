"""
Generate 6-fold participant split files for LODO evaluation
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

LABEL_FILES = {
    'BMCLab': 'assets/datasets/BMCLab.pkl',
    '3DGait': 'assets/datasets/3DGait.pkl',
    'PD-GaM': 'assets/datasets/PD-GaM.pkl',
    'T-SDU-PD': 'assets/datasets/T-SDU-PD.pkl',
}

OUTPUT_DIR = Path('assets/datasets/folds/UPDRS_Datasets')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_patient_labels(dataset_name):
    """Get patient IDs and their labels from dataset"""
    label_path = LABEL_FILES[dataset_name]
    with open(label_path, 'rb') as f:
        data = pickle.load(f)

    patient_labels = {}

    for patient_id, walks in data.items():
        if isinstance(walks, dict):
            for walk_id, walk_data in walks.items():
                if isinstance(walk_data, dict):
                    label = None
                    for key in ['UPDRS_GAIT', 'UPDRS_gait', 'updrs_gait', 'gait', 'GAIT']:
                        if key in walk_data:
                            val = walk_data[key]
                            if val is not None:
                                label = int(val)
                            break

                    if label is not None:
                        # Keep patient_id as string
                        patient_labels[str(patient_id)] = label
                        break

    return patient_labels

def generate_fold_file(dataset_name, n_folds=6):
    """Generate n-fold stratified split for a dataset"""
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}")
    print(f"{'='*50}")

    try:
        patient_labels = get_patient_labels(dataset_name)

        if not patient_labels:
            print(f"  No labels found for {dataset_name}")
            return False

        patients = sorted(patient_labels.keys())
        labels = np.array([patient_labels[p] for p in patients])

        print(f"  Patients: {len(patients)}")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique, counts))}")

        # Merge rare classes (3+) into class 2 for fold generation
        labels_for_split = labels.copy()
        labels_for_split[labels_for_split >= 3] = 2

        unique_split, counts_split = np.unique(labels_for_split, return_counts=True)
        min_count = min(counts_split)

        actual_folds = min(min_count, n_folds)
        if actual_folds < 2:
            actual_folds = 2
            print(f"  Warning: Forcing minimum 2 folds")

        print(f"  Using {actual_folds} folds")

        # Generate stratified folds
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        folds = {}
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patients, labels_for_split)):
            train_patients = [patients[i] for i in train_idx]
            eval_patients = [patients[i] for i in test_idx]
            # Use 'eval' key instead of 'test' to match expected format
            folds[fold_idx + 1] = {
                'train': train_patients,
                'eval': eval_patients
            }
            print(f"  Fold {fold_idx + 1}: train={len(train_patients)}, eval={len(eval_patients)}")

        # Save fold file
        output_path = OUTPUT_DIR / f"{dataset_name}_{n_folds}fold_participants.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(folds, f)

        print(f"  Saved: {output_path}")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Generating 6-fold participant splits for LODO evaluation")
    print("Using 'eval' key (not 'test') to match expected format")

    for dataset in LABEL_FILES.keys():
        generate_fold_file(dataset, n_folds=6)

    print("\n" + "="*50)
    print("Done! Check assets/datasets/folds/UPDRS_Datasets/")
    print("="*50)
