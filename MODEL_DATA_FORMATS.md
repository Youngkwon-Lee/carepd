# CARE-PD 모델별 데이터 형식 요구사항 (2026-01-08)

## 데이터 형식 종류

### 1. H36M (Human3.6M 17 Joints)
- **파일**: `h36m_3d_world_floorXZZplus_30f_or_longer.npz`
- **크기**: 83MB
- **형식**: [B, T, 17, 3] (17개 관절, 3D 좌표)
- **사용 모델**: POTR

### 2. H36M_backright (카메라 뷰 변환)
- **파일**: `h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz`
- **크기**: 165MB
- **형식**: World → Camera 좌표 변환 (backright 뷰)
- **사용 모델**: MotionBERT, PoseFormerV2, MixSTE, MotionAGFormer

### 3. H36M_sideright (카메라 뷰 변환)
- **파일**: `h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz`
- **크기**: 165MB
- **형식**: World → Camera 좌표 변환 (sideright 뷰)
- **사용 모델**: MotionBERT, PoseFormerV2, MixSTE, MotionAGFormer
- **용도**: Two-view 앙상블 (backright + sideright)

### 4. 6D_SMPL
- **파일**: `6D_SMPL_30f_or_longer.npz` (미생성)
- **형식**: SMPL 6D rotation representation
- **사용 모델**: MotionCLIP
- **상태**: ❌ 미전처리

### 5. HumanML3D
- **파일**: `HumanML3D_collected.npz` (미생성)
- **크기**: 예상 ~100-200MB
- **형식**: 263차원 feature representation
- **사용 모델**: MoMask (RVQVAE)
- **상태**: ❌ 미전처리

---

## 모델별 데이터 요구사항

| Model | Params | 데이터 형식 | 설정 파일 | 전처리 상태 | 학습 가능 |
|-------|--------|------------|-----------|------------|-----------|
| **POTR** | 3.3M | H36M | `BMCLab.json` | ✅ | ✅ **완료** |
| **MotionBERT** | 25M | H36M_backright | `BMCLab_backright.json` | ✅ | ✅ **준비됨** |
| **PoseFormerV2** | 8M | H36M_backright | `BMCLab_backright.json` | ✅ | ✅ **준비됨** |
| **MixSTE** | 10M | H36M_backright | `BMCLab_backright.json` | ✅ | ✅ **준비됨** |
| **MotionAGFormer** | 15M | H36M_backright | `BMCLab_backright.json` | ✅ | ✅ **준비됨** |
| **MotionCLIP** | 8M | 6D_SMPL | `BMCLab.json` | ❌ | ❌ 전처리 필요 |
| **MoMask** | 10M | HumanML3D | `BMCLab.json` | ❌ | ❌ 전처리 필요 |

---

## 전처리 스크립트

### H36M 형식 (이미 실행됨)
```bash
# 기본 H36M 전처리
python data/preprocessing/smpl2h36m.py -db "BMCLab"

# 또는 스크립트 사용
bash scripts/preprocess_smpl2h36m.sh
```

**결과:**
- `h36m_3d_world_floorXZZplus_30f_or_longer.npz`
- `h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz`
- `h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz`

### 6D_SMPL 형식 (미실행)
```bash
# 6D SMPL 전처리
bash scripts/preprocess_smpl2sixD.sh

# 또는 직접 실행
python data/preprocessing/smpl2sixD.py -db "BMCLab"
```

### HumanML3D 형식 (미실행)
```bash
# HumanML3D 전처리
bash scripts/preprocess_smpl2humanml3d.sh

# 또는 직접 실행
python data/preprocessing/smpl2humanml3d.py -db "BMCLab"
```

---

## Two-View 앙상블 전략

**Two-View 모델:**
- MotionBERT, PoseFormerV2, MixSTE, MotionAGFormer

**학습 방법:**
1. **Backright 뷰 학습**
   ```bash
   python run.py --backbone motionbert --config BMCLab_backright.json --num_folds 23 --hypertune 0
   ```

2. **Sideright 뷰 학습**
   ```bash
   python run.py --backbone motionbert --config BMCLab_sideright.json --num_folds 23 --hypertune 0
   ```

3. **예측 결합 (앙상블)**
   ```bash
   python run.py \
       --backbone motionbert \
       --num_folds -1 \
       --combine_views_preds 1 \
       --views_path \
           "Hypertune/motionbert_BMCLab_backright/0" \
           "Hypertune/motionbert_BMCLab_sideright/0"
   ```

**장점:**
- 두 카메라 뷰의 정보를 결합하여 성능 향상
- 논문에서도 사용한 공식 방법

---

## 학습 우선순위 (현재 상태 기준)

### 🟢 즉시 학습 가능 (H36M_backright 데이터 존재)

**우선순위 1: 경량 모델**
- [ ] **PoseFormerV2** (8M params)
  - POTR과 유사한 Transformer 구조
  - 빠른 학습 속도
  - 예상 시간: ~10-15분 (23-fold)

**우선순위 2: 중간 모델**
- [ ] **MixSTE** (10M params)
  - Spatio-Temporal Encoder
  - 예상 시간: ~15-20분

- [ ] **MotionAGFormer** (15M params)
  - Attention + Graph 하이브리드
  - 예상 시간: ~20-25분

**우선순위 3: 대형 모델**
- [ ] **MotionBERT** (25M params)
  - 가장 강력한 모델 (POTR의 7.5배)
  - Dual-Stream Transformer
  - 예상 시간: ~30-40분
  - **주의**: GPU 메모리 11GB+ 필요 → GPU 0 사용 또는 batch_size 조정

### 🟡 전처리 후 학습 가능

- [ ] **MotionCLIP** (8M params)
  - 6D_SMPL 전처리 필요
  - 전처리 시간: ~5-10분

- [ ] **MoMask** (10M params)
  - HumanML3D 전처리 필요
  - 전처리 시간: ~10-15분

---

## 코드 수정 사항 (완료)

### dataloaders.py 수정
- **파일**: `data/dataloaders.py`
- **수정 내용**: Line 181-183 주석처리
  ```python
  # elif self.params['dataset'] == 'BMCLab' and num_folds == 23:
  #     cv_folds = pickle.load(open(path.BMCLab_23FOLD_SPLIT, "rb"))
  #     print(f'Using vidas custom 23fold (LOSO) split for BMCLab data {path.BMCLab_23FOLD_SPLIT}')
  ```
- **이유**: 23-fold split 파일이 없어서 동적 생성되도록 수정
- **백업**: `data/dataloaders.py.backup`

---

## 다음 단계

### 1. 경량 모델 학습 (권장)
```bash
# PoseFormerV2 하이퍼튜닝
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --backbone poseformerv2 \
    --config BMCLab_backright.json \
    --num_folds 2 \
    --hypertune 1 \
    --ntrials 3 \
    --this_run_num 0 > poseformerv2_hypertune.log 2>&1 &
```

### 2. MotionBERT 학습 (성능 최고 기대)
```bash
# MotionBERT 하이퍼튜닝
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --backbone motionbert \
    --config BMCLab_backright.json \
    --num_folds 2 \
    --hypertune 1 \
    --ntrials 3 \
    --this_run_num 0 > motionbert_hypertune.log 2>&1 &
```

### 3. 추가 전처리 (선택)
```bash
# 6D_SMPL 전처리 (MotionCLIP용)
bash scripts/preprocess_smpl2sixD.sh

# HumanML3D 전처리 (MoMask용)
bash scripts/preprocess_smpl2humanml3d.sh
```

---

**마지막 업데이트**: 2026-01-08
**작성자**: Claude + YK
