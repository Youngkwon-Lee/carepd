#!/bin/bash
# CARE-PD HPC Environment Setup Script
# Run this on HPC after transferring files

set -e

echo "=========================================="
echo "CARE-PD HPC Environment Setup"
echo "=========================================="

# Check CUDA
echo "[1/5] Checking CUDA..."
nvidia-smi || { echo "ERROR: nvidia-smi failed. Check GPU access."; exit 1; }

# Create conda environment
echo "[2/5] Creating conda environment (Python 3.9)..."
conda create -n carepd python=3.9 -y

# Activate environment
echo "[3/5] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate carepd

# Install PyTorch with CUDA 12.1 (compatible with HPC CUDA 12.4)
echo "[4/5] Installing PyTorch..."
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install CARE-PD dependencies
echo "[5/5] Installing CARE-PD dependencies..."
cd ~/CARE-PD

# Core dependencies (excluding chumpy which has build issues)
pip install \
    einops>=0.6.0 \
    optuna>=3.3.0 \
    smplx>=0.1.28 \
    timm>=0.9.2 \
    torch_dct>=0.1.6 \
    trimesh>=4.0.0 \
    wandb>=0.15.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    scipy>=1.11.0 \
    pandas>=2.0.0 \
    h5py>=3.9.0 \
    pyyaml>=6.0

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    if torch.cuda.device_count() > 1:
        print(f'GPU 1: {torch.cuda.get_device_name(1)}')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. conda activate carepd"
echo "  2. cd ~/CARE-PD"
echo "  3. python run.py --backbone motionbert --dataset BMCLab"
echo ""
