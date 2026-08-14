import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path

class MergedReader():
    """Reads merged CARE-PD dataset from all 9 cohorts."""

    DATASETS_WITH_UPDRS = ['BMCLab', '3DGait', 'PD-GaM', 'T-SDU-PD', 'DNE']

    def __init__(self, merged_npz_path, labels_dir, params):
        # Handle list input from dataloaders.py
        self.merged_npz_path = merged_npz_path[0] if isinstance(merged_npz_path, list) else merged_npz_path
        self.labels_dir = Path(labels_dir)
        self.params = params
        self.label_dfs = self._load_all_labels()
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict, self.medication_dict, self.FoG_labels_dict = self.read_keypoints_and_labels()
        print(f"Loaded {len(self.pose_dict)} sequences from merged dataset.")
        print(f"Total {len(set(self.participant_ID))} unique participants.")
        unique, counts = np.unique(list(self.labels_dict.values()), return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")

    def _load_all_labels(self):
        label_dfs = {}
        for ds in self.DATASETS_WITH_UPDRS:
            pkl_path = self.labels_dir / f'{ds}.pkl'
            if pkl_path.exists():
                label_dfs[ds] = joblib.load(pkl_path)
                print(f"  Loaded labels from {ds}.pkl")
        return label_dfs

    def _parse_seq_name(self, merged_key):
        parts = merged_key.split('_', 1)
        dataset = parts[0]
        original_key = parts[1] if len(parts) > 1 else ''
        return dataset, original_key

    def _get_label(self, dataset, original_key):
        if dataset not in self.label_dfs:
            return None, None, None
        try:
            label_df = self.label_dfs[dataset]
            parts = original_key.split('__')
            subject_id = parts[0]
            walkid = parts[1].split('_down')[0] if len(parts) > 1 else parts[0]
            if subject_id in label_df and walkid in label_df[subject_id]:
                entry = label_df[subject_id][walkid]
                label = entry.get('UPDRS_GAIT')
                med = entry.get('medication')
                fog = entry.get('other')
                if label is not None:
                    return int(label), med, fog
        except:
            pass
        return None, None, None

    def read_keypoints_and_labels(self):
        pose_dict = {}
        labels_dict = {}
        metadata_dict = {}
        fog_dict = {}
        medication_dict = {}
        video_names_list = []
        participant_ID = []

        print(f'Loading merged dataset from {self.merged_npz_path}')
        seqs = np.load(self.merged_npz_path, allow_pickle=True)

        skipped = 0
        for merged_key in tqdm(seqs.files):
            dataset, original_key = self._parse_seq_name(merged_key)
            label, med, fog = self._get_label(dataset, original_key)

            if label is None:
                skipped += 1
                continue

            joints = seqs[merged_key]
            if joints.ndim == 2:
                joints = np.expand_dims(joints, axis=0)
            else:
                joints = joints[:1, ...]

            subject_id = f"{dataset}_{original_key.split('__')[0]}"

            pose_dict[merged_key] = joints
            labels_dict[merged_key] = label
            medication_dict[merged_key] = med
            fog_dict[merged_key] = fog
            metadata_dict[merged_key] = {}
            video_names_list.append(merged_key)
            participant_ID.append(subject_id)

        print(f"Skipped {skipped} sequences without UPDRS labels")
        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict, medication_dict, fog_dict
