# CARE-PD 프로젝트 현황 (2026-01-08)

## 📊 프로젝트 개요

**목표**: CARE-PD 데이터셋 기반 파킨슨병 보행 평가 모델 학습
**GitHub**: https://github.com/Youngkwon-Lee/carepd
**HPC 환경**: gun3856@10.246.246.111 (vmgnode47.openhpc.prv)

---

## ✅ 완료된 작업

### 1. 환경 구축 (2026-01-06)
- [x] GitHub 레포지토리 초기화
- [x] 로컬 레포지토리 구성: `C:\Users\YK\carepd`
- [x] HPC conda 환경 생성: `carepd` (Python 3.9)
- [x] PyTorch 설치: 2.1.0+cu121 (CUDA 12.1)
- [x] 필수 패키지 설치: einops, optuna, smplx, timm, wandb 등
- [x] NumPy 버전 조정: 1.26.4 (2.0 호환성 문제)
- [x] chumpy 패치: NumPy deprecated aliases 수정

### 2. 데이터 준비
- [x] HuggingFace에서 데이터셋 다운로드 (D:\carepd)
- [x] WinSCP로 HPC 전송: `~/carepd/CARE-PD/assets/datasets/`
  - BMCLab.pkl (122MB)
  - 3DGait.pkl, PD-GaM.pkl, T-SDU-PD.pkl 등
  - folds/ 디렉토리 (fold 인덱스)
- [x] 전처리 파일 전송: `data/preprocessing/common/`
  - J_regressor_h36m_correct.npy
  - body_models/smpl/SMPL_NEUTRAL.pkl (39MB)

### 3. 데이터 전처리
- [x] **BMCLab**: SMPL → H36M 변환 완료
  - 총 3,895 시퀀스 생성
  - 저장 위치: `~/carepd/CARE-PD/assets/datasets/h36m/BMCLab/`
- [ ] 3DGait: 전처리 스크립트 버그 (UnboundLocalError)
- [ ] PD-GaM: 미실행
- [ ] T-SDU-PD: 미실행

### 4. 모델 학습
- [x] 사전학습 모델 다운로드: `bash scripts/download_models.sh`
- [x] **POTR + BMCLab 하이퍼파라미터 튜닝 완료** (2026-01-08)
  - GPU: Tesla V100 (GPU 1)
  - Trials: 3
  - **Best F1 Score: 0.4036 (40.36%)**
  - Best Trial: 0
  - 최적 설정:
    - epochs: 30
    - batch_size: 256
    - optimizer: AdamW
    - lr_backbone: 0.0001
    - lr_head: 0.001
    - criterion: FocalLoss (alpha=1, gamma=2)
    - weight_decay: 0
  - 결과 저장: `./experiment_outs/Hypertune/POTR_BMCLab/0/study.pkl`

---

## 🖥️ HPC 환경 상세

### GPU 현황
| GPU | 모델 | 메모리 사용 | 상태 | 비고 |
|-----|------|-------------|------|------|
| GPU 0 | Tesla V100-PCIE-16GB | 14.2GB/16GB | **사용 중** | med-prm-vl 프로세스 (PID: 1063049, 1063158) - **건드리면 안됨** |
| GPU 1 | Tesla V100-PCIE-16GB | ~4MB/16GB | **사용 가능** | POTR 학습에 사용 |

### 환경 정보
```bash
Host: vmgnode47.openhpc.prv
IP: 10.246.246.111
User: gun3856
만료일: 2025-12-31

# 접속
ssh gun3856@10.246.246.111

# conda 환경 활성화
conda activate carepd

# 작업 디렉토리
cd ~/carepd/CARE-PD
```

### 설치된 패키지 (주요)
```
Python: 3.9.21
torch: 2.1.0+cu121
numpy: 1.26.4
einops, optuna, smplx, timm, wandb
scikit-learn, tqdm, matplotlib, scipy, pandas
```

---

## 📁 파일 구조

### 로컬 (C:\Users\YK\carepd)
```
carepd/
├── CARE-PD/                    # 소스코드
│   ├── model/                  # 7개 모델 구현
│   ├── data/                   # 데이터 로더, 전처리
│   ├── configs/                # 모델별 설정 파일
│   ├── hpc/                    # HPC 배포 스크립트
│   └── run.py                  # 메인 학습 스크립트
├── CARE-PD_DATASET.md          # 데이터셋 문서
├── MODEL_ARCHITECTURES.md      # 모델 아키텍처 분석
├── DATA_PIPELINE_ANALYSIS.md   # 데이터 파이프라인 분석
└── .gitignore                  # 대용량 파일 제외
```

### HPC (~/carepd/CARE-PD)
```
CARE-PD/
├── assets/
│   ├── datasets/
│   │   ├── BMCLab.pkl (122MB)
│   │   ├── 3DGait.pkl, PD-GaM.pkl, T-SDU-PD.pkl
│   │   ├── folds/ (fold 인덱스)
│   │   └── h36m/
│   │       └── BMCLab/ (전처리 완료, 3895 시퀀스)
│   └── Pretrained_checkpoints/
│       ├── potr/pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt
│       ├── motionbert/motionbert.bin
│       └── momask/, motionclip/, motionagformer/, mixste/, poseformerv2/
└── experiment_outs/
    └── Hypertune/
        └── POTR_BMCLab/
            └── 0/study.pkl (튜닝 결과)
```

---

## 📝 다음 작업 계획

### 우선순위 1: 나머지 경량 모델 학습 (GPU 1 사용)
- [ ] **MotionCLIP** (8M params) + BMCLab
- [ ] **PoseFormerV2** (8M params) + BMCLab
- [ ] **MixSTE** (10M params) + BMCLab
- [ ] **MoMask (RVQVAE)** (10M params) + BMCLab

### 우선순위 2: 추가 데이터셋 전처리
- [ ] 3DGait 전처리 버그 수정
- [ ] PD-GaM 전처리
- [ ] T-SDU-PD 전처리

### 우선순위 3: 중량 모델 학습 (GPU 0 사용 가능 시)
- [ ] **MotionBERT** (25M params) - GPU 0 필요 (11GB+ 메모리)
- [ ] **MotionAGFormer** (15M params)

### 우선순위 4: 평가 및 분석
- [ ] LOSO (Leave-One-Subject-Out) Cross-Validation
- [ ] Cross-Dataset Evaluation
- [ ] 모델별 성능 비교 분석

---

## ⚠️ 주의사항

1. **GPU 0 프로세스 보호**
   - PID 1063049, 1063158 (med-prm-vl) 절대 kill 금지
   - GPU 0 사용 전 메모리 확인 필수

2. **파일 전송**
   - 대용량 파일(.pkl, .pth)은 WinSCP 사용
   - 코드만 Git으로 관리

3. **환경 활성화**
   ```bash
   conda activate carepd
   export WANDB_MODE=disabled  # wandb 사용 안할 시
   ```

4. **학습 실행 예시**
   ```bash
   # GPU 1에서 POTR 학습
   CUDA_VISIBLE_DEVICES=1 python run.py \
       --backbone potr \
       --config BMCLab.json \
       --num_folds 2 \
       --hypertune 1 \
       --ntrials 3 \
       --this_run_num 0
   ```

---

## 🔗 참고 링크

- **Original Repo**: https://github.com/TaatiTeam/CARE-PD
- **My Fork**: https://github.com/Youngkwon-Lee/carepd
- **HuggingFace Dataset**: https://huggingface.co/datasets/vida-adl/CARE-PD
- **Paper**: NeurIPS 2025 (Datasets & Benchmarks Track)

---

## 📊 학습 결과 요약

| Model | Dataset | Params | F1 Score | Trials | GPU | Status |
|-------|---------|--------|----------|--------|-----|--------|
| POTR | BMCLab | 3.3M | **0.4036** | 3 | GPU 1 | ✅ 완료 |
| MotionCLIP | BMCLab | 8M | - | - | - | ⏳ 대기 |
| PoseFormerV2 | BMCLab | 8M | - | - | - | ⏳ 대기 |
| MixSTE | BMCLab | 10M | - | - | - | ⏳ 대기 |
| MoMask | BMCLab | 10M | - | - | - | ⏳ 대기 |
| MotionAGFormer | BMCLab | 15M | - | - | - | ⏳ 대기 |
| MotionBERT | BMCLab | 25M | - | - | GPU 0 필요 | ⏳ 대기 |

---

**마지막 업데이트**: 2026-01-08
**작성자**: Claude + YK
**다음 세션 시작 시**: 이 파일 읽고 바로 작업 이어가기
