import numpy as np
from pathlib import Path

DATASETS = {
    'BMCLab': 'h36m_3d_world_floorXZZplus_30f_or_longer.npz',
    '3DGait': 'h36m_3d_world_floorXZZplus_30f_or_longer.npz',
    'PD-GaM': 'h36m_3d_world_floorXZZplus_30f_or_longer.npz',
    'T-SDU-PD': 'h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz',
    'DNE': 'h36m_3d_world_floorXZZplus_30f_or_longer.npz',
    'E-LC': 'h36m_3d_world_floorXZZplus_30f_or_longer.npz',
    'KUL-DT-T': 'h36m_3d_world_floorXZZplus_30f_or_longer.npz',
    'T-LTC': 'h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz',
    'T-SDU': 'h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz',
}

BASE_PATH = Path('assets/datasets/h36m')
OUT_DIR = BASE_PATH / 'merged'
OUT_DIR.mkdir(exist_ok=True)

merged = {}
total = 0

for ds, fname in DATASETS.items():
    fpath = BASE_PATH / ds / fname
    if not fpath.exists():
        print(f"SKIP {ds}: {fpath} not found")
        continue
    data = np.load(fpath, allow_pickle=True)
    count = 0
    for key in data.files:
        new_key = f"{ds}_{key}"
        merged[new_key] = data[key]
        count += 1
    print(f"{ds}: {count} sequences")
    total += count

out_path = OUT_DIR / 'CARE-PD_merged_3d_preprocessed.npz'
np.savez(out_path, **merged)
print(f"\nTotal: {total} sequences saved to {out_path}")
