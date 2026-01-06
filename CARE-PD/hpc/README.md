# CARE-PD HPC Deployment

> HPC GPU 클러스터에서 CARE-PD 모델 학습

## HPC 환경 정보

| 항목 | 값 |
|------|-----|
| **Host** | vmgnode47.openhpc.prv |
| **IP** | 10.246.246.111 |
| **User** | gun3856 |
| **GPU** | NVIDIA V100 16GB x 2 |
| **CUDA** | 12.4 |
| **만료일** | 2025-12-31 |

## 빠른 시작

### Step 1: 로컬에서 파일 전송

```bash
# CARE-PD 코드 전송
cd C:/Users/YK/carepd
scp -r CARE-PD gun3856@10.246.246.111:~/

# 데이터셋 전송 (D:\carepd → HPC)
scp -r /d/carepd/*.pkl gun3856@10.246.246.111:~/CARE-PD/assets/datasets/
scp -r /d/carepd/folds gun3856@10.246.246.111:~/CARE-PD/assets/datasets/
```

### Step 2: HPC 접속 및 환경 설정

```bash
# SSH 접속
ssh gun3856@10.246.246.111

# conda 환경 생성
cd ~/CARE-PD
bash hpc/setup_env.sh

# 환경 활성화
conda activate carepd
```

### Step 3: GPU 확인

```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Step 4: 학습 실행

```bash
# 단일 데이터셋 학습 (예: BMCLab + MotionBERT)
cd ~/CARE-PD
nohup python run.py --backbone motionbert --dataset BMCLab --num_folds 6 > train_motionbert_bmclab.log 2>&1 &

# 로그 모니터링
tail -f train_motionbert_bmclab.log
```

## 학습 옵션

### Backbone 선택

| Backbone | 권장 Batch Size | GPU 메모리 |
|----------|-----------------|-----------|
| `motionbert` | 32 | ~8 GB |
| `potr` | 64 | ~4 GB |
| `momask` | 32 | ~6 GB |
| `motionagformer` | 32 | ~6 GB |
| `motionclip` | 32 | ~5 GB |

### 데이터셋 선택

| Dataset | UPDRS 점수 | 설명 |
|---------|-----------|------|
| `3DGait` | O | 43명 환자 |
| `BMCLab` | O | 23명 환자 |
| `PD-GaM` | O | 30명 환자 |
| `T-SDU-PD` | O | 14명 환자 |

### 예시 명령어

```bash
# MotionBERT + BMCLab (6-fold CV)
python run.py --backbone motionbert --dataset BMCLab --num_folds 6

# POTR + 3DGait (LOSO)
python run.py --backbone potr --dataset 3DGait --num_folds 43

# Multi-GPU 학습
CUDA_VISIBLE_DEVICES=0,1 python run.py --backbone motionbert --dataset BMCLab
```

## 결과 다운로드

```bash
# 로컬에서 실행
scp -r gun3856@10.246.246.111:~/CARE-PD/out ./results_hpc/
scp gun3856@10.246.246.111:~/CARE-PD/*.log ./logs/
```

## 예상 학습 시간

| 설정 | V100 1개 | V100 2개 |
|------|---------|---------|
| MotionBERT + BMCLab (6-fold) | ~2시간 | ~1시간 |
| MotionBERT + 3DGait (LOSO 43-fold) | ~8시간 | ~4시간 |
| 전체 실험 (7 backbone × 4 dataset) | ~56시간 | ~28시간 |

## 문제 해결

### OOM (Out of Memory)
```bash
# batch size 줄이기
python run.py --backbone motionbert --batch_size 16
```

### CUDA 버전 불일치
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 필요시 재설치
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## 파일 구조

```
~/CARE-PD/
├── hpc/
│   ├── README.md          # 이 파일
│   ├── setup_env.sh       # conda 환경 설정
│   └── train_all.sh       # 전체 실험 스크립트
├── assets/datasets/       # 데이터셋 (전송 필요)
│   ├── *.pkl
│   └── folds/
├── run.py                 # 메인 학습 스크립트
├── train.py               # 학습 함수
└── out/                   # 결과 저장
```
