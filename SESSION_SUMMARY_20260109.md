# CARE-PD 프로젝트 세션 요약 (2026-01-09)

## 🎯 완료된 작업

### 1. POTR 모델 학습 완료
- ✅ 하이퍼파라미터 튜닝 (3 trials, Best F1: 0.4036)
- ✅ 2-fold LOSO (Weighted F1: 0.40)
- ✅ **23-fold LOSO 완료** (Weighted F1: 0.42)
  - **핵심 성과**: Class 2 (중증) F1 Score: 0.03 → **0.25** (+733%!)
  - 중증 환자 탐지율: 3% → 27% (9배 향상)

### 2. 데이터 전처리 완료
- ✅ BMCLab SMPL → H36M 변환 (5개 파일, 525MB)
  - `h36m_3d_world_floorXZZplus_30f_or_longer.npz` (83MB)
  - `h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz` (165MB)
  - `h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz` (165MB)
  - backright/sideright 이미지 좌표 변환 파일 2개 (각 56MB)

### 3. 백업 파일 생성
- ✅ `potr_23fold_results_20260108.tar.gz` (273MB) - 23-fold 모델
- ✅ `potr_all_results_20260108.tar.gz` (299MB) - 전체 결과
- **위치**: `~/carepd/CARE-PD/`

### 4. 문서화 완료
- ✅ `STATUS.md` - 전체 프로젝트 현황
- ✅ `RESULTS_ANALYSIS.md` - POTR 결과 상세 분석
- ✅ `MODEL_DATA_FORMATS.md` - 모델별 데이터 형식 가이드
- ✅ Git 커밋 & Push 완료 (GitHub에 저장)

### 5. 코드 수정
- ✅ `data/dataloaders.py` Line 181-183 주석처리
  - 23-fold split 동적 생성 가능하도록 수정
  - 백업: `data/dataloaders.py.backup`

---

## 🔍 학습 시도 및 실패 분석

### MotionCLIP (실패 - 데이터 형식 불일치)
- **에러**: `FileNotFoundError: 6D_SMPL_30f_or_longer.npz`
- **원인**: 6D_SMPL 형식 데이터 미전처리
- **해결**: `bash scripts/preprocess_smpl2sixD.sh` 실행 필요

### MoMask (실패 - 데이터 형식 불일치)
- **에러**: `FileNotFoundError: HumanML3D_collected.npz`
- **원인**: HumanML3D 형식 데이터 미전처리
- **해결**: `bash scripts/preprocess_smpl2humanml3d.sh` 실행 필요

### PoseFormerV2 (부분 실패 - 설정 파일 오류 + X11 에러)
- **에러 1**: `BMCLab.json` 사용 (잘못됨, `BMCLab_backright.json` 필요)
- **에러 2**: `XIO: fatal IO error 22` (X11 디스플레이 문제)
- **진행 상황**: Fold 2의 10 epochs 학습 완료 시점에 중단
  - Train accuracy: 85.8%, Val F1: 0.599
- **해결**:
  1. 올바른 설정 파일 사용: `--config BMCLab_backright.json`
  2. X11 문제 방지: `export MPLBACKEND=Agg` 추가

---

## 📊 POTR 최종 결과 (23-Fold LOSO)

### 전체 성능
```
Overall Accuracy: 43%
Weighted F1 Score: 0.42
Macro F1 Score: 0.39
Total Samples: 3,895
```

### 클래스별 성능
| Class | Precision | Recall | F1-Score | Support | 의미 |
|-------|-----------|--------|----------|---------|------|
| 0 | 0.53 | 0.59 | **0.56** | 1,705 | 정상 (양호) |
| 1 | 0.41 | 0.32 | **0.36** | 1,380 | 경증 (보통) |
| 2 | 0.24 | 0.27 | **0.25** | 810 | 중증 (개선됨!) |

### 2-Fold vs 23-Fold 비교
| 지표 | 2-Fold | 23-Fold | 변화 |
|------|--------|---------|------|
| Weighted F1 | 0.40 | **0.42** | +5% |
| Macro F1 | 0.33 | **0.39** | +18% |
| Class 2 F1 | 0.03 | **0.25** | +733% 🚀 |

---

## 🚀 다음 세션 작업 계획

### 우선순위 1: H36M_backright 모델 학습 (즉시 가능)

#### PoseFormerV2 (8M params - 권장)
```bash
cd ~/carepd/CARE-PD

# X11 에러 방지
export MPLBACKEND=Agg

# 하이퍼튜닝 (올바른 설정 파일 사용!)
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --backbone poseformerv2 \
    --config BMCLab_backright.json \
    --num_folds 2 \
    --hypertune 1 \
    --ntrials 3 \
    --this_run_num 0 > poseformerv2_backright_hypertune.log 2>&1 &

# 로그 확인
tail -f poseformerv2_backright_hypertune.log
```

#### MotionBERT (25M params - 최강)
```bash
# X11 에러 방지
export MPLBACKEND=Agg

# 하이퍼튜닝
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --backbone motionbert \
    --config BMCLab_backright.json \
    --num_folds 2 \
    --hypertune 1 \
    --ntrials 3 \
    --this_run_num 0 > motionbert_backright_hypertune.log 2>&1 &
```

**예상 시간:**
- PoseFormerV2: ~10-15분
- MotionBERT: ~30-40분 (메모리 주의)

### 우선순위 2: 추가 전처리 (선택)

#### 6D_SMPL (MotionCLIP용)
```bash
cd ~/carepd/CARE-PD
bash scripts/preprocess_smpl2sixD.sh
```

#### HumanML3D (MoMask용)
```bash
cd ~/carepd/CARE-PD
bash scripts/preprocess_smpl2humanml3d.sh
```

### 우선순위 3: 베이스라인 비교
- 논문의 공식 베이스라인 결과 찾기
- POTR 결과와 비교 분석

---

## ⚠️ 주의사항

### GPU 사용
- **GPU 0**: med-prm-vl 프로세스 실행 중 (PID: 1063049, 1063158)
  - **절대 건드리면 안됨!**
  - 14.2GB/16GB 사용 중
- **GPU 1**: 사용 가능 (~4MB 사용)
  - 모든 학습에 GPU 1 사용: `CUDA_VISIBLE_DEVICES=1`

### X11 에러 방지
```bash
# 학습 시작 전 반드시 실행!
export MPLBACKEND=Agg
```

### 설정 파일 확인
| Model | 올바른 설정 파일 | 잘못된 설정 파일 |
|-------|-----------------|-----------------|
| POTR | `BMCLab.json` | - |
| MotionBERT | `BMCLab_backright.json` | `BMCLab.json` ❌ |
| PoseFormerV2 | `BMCLab_backright.json` | `BMCLab.json` ❌ |
| MixSTE | `BMCLab_backright.json` | `BMCLab.json` ❌ |
| MotionAGFormer | `BMCLab_backright.json` | `BMCLab.json` ❌ |

---

## 📁 중요 파일 위치

### HPC (`~/carepd/CARE-PD/`)
```
백업 파일:
├── potr_23fold_results_20260108.tar.gz (273MB)
├── potr_all_results_20260108.tar.gz (299MB)
└── potr_2fold_results_20260108.tar.gz (24MB)

데이터:
└── assets/datasets/h36m/BMCLab/*.npz (5개 파일, 525MB)

로그:
├── train_potr_23fold.log (완료)
├── poseformerv2_hypertune.log (중단됨)
└── motionbert_backright_hypertune.log (미완료)

코드 수정:
└── data/dataloaders.py.backup (백업)
```

### GitHub
- Repository: https://github.com/Youngkwon-Lee/carepd
- Commit: `1f88fec`
- 모든 문서 동기화됨

---

## 🎯 빠른 시작 (다음 세션)

### 1. HPC 접속 및 환경 준비
```bash
ssh gun3856@10.246.246.111
conda activate carepd
cd ~/carepd/CARE-PD
```

### 2. 상태 확인
```bash
# GPU 상태
nvidia-smi

# 백업 파일
ls -lh *.tar.gz

# 실행 중인 프로세스
ps aux | grep python | grep run.py
```

### 3. PoseFormerV2 학습 시작 (권장)
```bash
export MPLBACKEND=Agg

CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --backbone poseformerv2 \
    --config BMCLab_backright.json \
    --num_folds 2 \
    --hypertune 1 \
    --ntrials 3 \
    --this_run_num 0 > poseformerv2_backright_hypertune.log 2>&1 &

tail -f poseformerv2_backright_hypertune.log
```

---

## 📚 참고 문서

### 로컬 (Git)
- `STATUS.md` - 전체 프로젝트 현황
- `RESULTS_ANALYSIS.md` - POTR 결과 분석
- `MODEL_DATA_FORMATS.md` - 모델별 데이터 형식
- `MODEL_ARCHITECTURES.md` - 모델 아키텍처 분석
- `DATA_PIPELINE_ANALYSIS.md` - 데이터 파이프라인

### HPC
- `~/carepd/CARE-PD/README.md` - 원본 레포 README
- `~/carepd/CARE-PD/scripts/` - 전처리 및 평가 스크립트

---

**마지막 업데이트**: 2026-01-09
**작성자**: Claude + YK
**세션 상태**: POTR 완료, 다음 모델 준비됨
