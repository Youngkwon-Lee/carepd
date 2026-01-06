# CARE-PD Data Pipeline Analysis

> 데이터 전처리, 증강, 학습 파이프라인 상세 분석

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CARE-PD Data Pipeline                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Raw SMPL Data]  ──→  [Preprocessing]  ──→  [DataLoader]  ──→  [Training]   │
│       .pkl              smpl2h36m.py         dataloaders.py    train.py      │
│                         smpl2humanml3d.py                                    │
│                         smpl2sixD.py                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Raw Data Format (SMPL)

### 원본 데이터 구조
```python
# *.pkl 파일 구조
{
    "subject_id": {
        "walk_id": {
            "pose": np.array,      # (num_frames, 24, 3) - SMPL pose params
            "trans": np.array,     # (num_frames, 3) - translation
            "beta": np.array,      # (10,) or (num_frames, 10) - body shape
            "fps": int,            # 프레임률
            "UPDRS_GAIT": int,     # 0-3 임상 점수 (또는 None)
            "medication": str,     # 'on'/'off' (또는 None)
            "other": str           # 추가 레이블
        }
    }
}
```

---

## 2. Preprocessing Pipeline

### 2.1 SMPL → H36M 변환 (smpl2h36m.py)

**핵심 프로세스:**
```
SMPL pose params → SMPL model → Vertices → H36M Joint Regressor → 17 joints
```

**주요 단계:**
1. **SMPL 모델 적용**: pose, beta, trans로 vertices 생성
2. **Joint Regression**: H36M regressor로 17개 관절 추출
3. **Slope Correction**: 걷기 방향 보정 (일부 데이터셋)
4. **Floor Placement**: Y축 최소값을 0으로 조정
5. **Origin Alignment**: 첫 프레임 XZ 위치를 원점으로
6. **Direction Normalization**: Z+ 방향으로 정렬

**카메라 뷰 생성:**
```python
views = ['back', 'front', 'sideleft', 'sideright', 'backright']
# World → Camera → Image 좌표 변환
```

### 2.2 SMPL → HumanML3D 변환 (smpl2humanml3d.py)

- 263차원 feature vector 생성
- 속도, 관절 위치, 회전 정보 포함

### 2.3 SMPL → 6D Rotation (smpl2sixD.py)

- 연속적인 6D 회전 표현 변환
- MotionCLIP 등에서 사용

---

## 3. Data Loader System

### 3.1 Backbone별 Preprocessor

| Backbone | Preprocessor | 특징 |
|----------|-------------|------|
| **POTR** | POTRPreprocessor | center + normalize |
| **MotionBERT** | MotionBERTPreprocessor | crop_scale 적용 |
| **MoMask** | MoMaskPreprocessor | 사전계산 증강 |
| **MotionAGFormer** | MotionAGFormerPreprocessor | crop_scale 적용 |
| **PoseformerV2** | PoseformerV2Preprocessor | screen coord normalize |
| **MixSTE** | MixSTEPreprocessor | screen coord normalize |
| **MotionCLIP** | MotionCLIPPreprocessor | 6D rotation |

### 3.2 Clipping Strategy

```python
def get_clips(video_sequence, clip_length):
    """비디오를 고정 길이 클립으로 분할"""

    if video_length < clip_length:
        # 패딩 적용
        clips.append(np.pad(video_sequence, ...))
        pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])
    else:
        if select_middle:
            # 중간 부분만 선택
            middle_frame = video_length // 2
            start_frame = middle_frame - (clip_length // 2)
        else:
            # 슬라이딩 윈도우
            while (video_length - start_frame) >= clip_length:
                clips.append(video_sequence[start_frame:start_frame + clip_length])
```

### 3.3 Cross-Validation Fold 생성

```python
# LOSO (Leave-One-Subject-Out) 또는 StratifiedKFold
fold_gen = LeaveOneOut().split(X) if num_folds == len(X)
           else StratifiedKFold(n_splits=num_folds).split(X, y)
```

---

## 4. Data Augmentation

### 4.1 Runtime Augmentations (augmentations.py)

| 증강 | 클래스 | 파라미터 | 설명 |
|------|--------|----------|------|
| **Mirror** | MirrorReflection | - | X축 반전 + 좌우 관절 교환 |
| **Rotation** | RandomRotation | min/max_rotate | 2D/3D 랜덤 회전 |
| **Noise** | RandomNoise | mean, std | 가우시안 노이즈 추가 |
| **Axis Mask** | axis_mask | - | 랜덤 축 제로화 |

### 4.2 Transform Pipeline

```python
runtime_train_transform = transforms.Compose([
    PreserveKeysTransform(transforms.RandomApply([
        MirrorReflection(data_dim=params['in_data_dim'])
    ], p=params['mirror_prob'])),
    PreserveKeysTransform(transforms.RandomApply([
        RandomRotation(*params['rotation_range'], data_dim=params['in_data_dim'])
    ], p=params['rotation_prob'])),
    PreserveKeysTransform(transforms.RandomApply([
        RandomNoise(data_dim=params['in_data_dim'], std=params['noise_std'])
    ], p=params['noise_prob'])),
    PreserveKeysTransform(transforms.RandomApply([
        axis_mask(data_dim=params['in_data_dim'])
    ], p=params['axis_mask_prob']))
])
```

---

## 5. Training Pipeline

### 5.1 run.py 구조

```python
# 1. Config 생성
config_generators = {
    'motionbert': generate_config_motionbert,
    'potr': generate_config_potr,
    'momask': generate_config_momask,
    # ...
}

# 2. 하이퍼파라미터 튜닝 (Optuna)
study = optuna.create_study(direction='maximize')
study.optimize(objective_hyperparam_opt_CV, n_trials=n_trials)

# 3. 학습 실행
train_model(model, train_loader, eval_loader, params, ...)
```

### 5.2 train.py 핵심 함수

```python
def train_model(model, train_loader, validation_loader, params, ...):
    """메인 학습 루프"""
    optimizer = choose_optimizer(params, model)
    scheduler = choose_scheduler(params, optimizer)
    criterion = choose_criterion(params['criterion'], params, class_weights)

    for epoch in range(params['epochs']):
        # Training step
        model.train()
        for x, y, video_idx, metadata, valid_mask in train_loader:
            out = model(x, metadata, valid_mask=valid_mask)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation step
        val_metrics = validate_model(model, validation_loader, ...)

def validate_model(model, validation_loader, params, class_weights):
    """검증 및 메트릭 계산"""
    # 클립별 예측 → 비디오별 집계 (majority vote)
    video_predictions = defaultdict(list)
    for x, y, video_idx, metadata, valid_mask in validation_loader:
        out = model(x, metadata, valid_mask=valid_mask)
        video_predictions[video_idx.item()].append(out)

    # Majority voting
    for video_idx in video_predictions:
        label_counts = Counter(label_predictions)
        video_prediction_label, _ = label_counts.most_common(1)[0]
```

### 5.3 Evaluation Metrics

- **Accuracy**: 정확도
- **F1 Score**: Macro/Weighted F1
- **Confusion Matrix**: 클래스별 혼동 행렬
- **MAE**: Mean Absolute Error (UPDRS 점수)

---

## 6. Key Constants & Paths

### 6.1 Joint Indices

```python
_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # H36M 17개
_MAJOR_JOINTS_MIRRORED = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
_HUMAN_ML3D_POSE_ELEMENTS = list(range(263))  # HumanML3D 263차원
_SMPL_6D_ELEMENTS = list(range(25))  # SMPL 25개 관절
```

### 6.2 Metadata Map

```python
METADATA_MAP = {
    'gender': 0,
    'age': 1,
    'height': 2,
    'weight': 3,
    'bmi': 4
}
```

---

## 7. 데이터셋별 특성

| Dataset | FPS | Slope Correction | Views |
|---------|-----|------------------|-------|
| **3DGait** | 30 | No | sideright, backright |
| **BMCLab** | 30 | No | sideright, backright |
| **T-SDU-PD** | 30 | Yes | sideright, backright |
| **PD-GaM** | 30 | No | sideright, backright |
| **DNE** | - | No | - |
| **E-LC** | - | No | - |
| **KUL-DT-T** | - | No | - |
| **T-LTC** | - | Yes | - |
| **T-SDU** | - | Yes | - |

---

## 8. PhysioKorea 통합 포인트

### 8.1 재사용 가능한 컴포넌트

1. **전처리 파이프라인**
   - `smpl2h36m.py`: SMPL → H36M 변환
   - `trajectory_correction.py`: 걷기 방향 보정

2. **데이터 증강**
   - `augmentations.py`: MirrorReflection, RandomRotation, RandomNoise

3. **학습 유틸리티**
   - Optuna 하이퍼파라미터 튜닝
   - WandB 실험 추적
   - Majority voting 기반 비디오 레벨 예측

### 8.2 통합 시나리오

```python
# PhysioKorea MLOps에서 CARE-PD 모델 활용
from care_pd.model.backbone_loader import load_pretrained_backbone
from care_pd.model.motion_encoder import MotionEncoder
from care_pd.data.augmentations import MirrorReflection, RandomRotation

# 보행 분석 파이프라인
class GaitAnalysisPipeline:
    def __init__(self):
        self.backbone = load_pretrained_backbone(params, 'motionbert')
        self.model = MotionEncoder(self.backbone, params, num_classes=4)

    def analyze(self, video_path):
        # MediaPipe로 포즈 추출 → CARE-PD 모델 추론
        poses = extract_poses(video_path)
        updrs_score = self.model.predict(poses)
        return updrs_score
```

---

*분석일: 2025-01-06*
*CARE-PD Repository: C:\Users\YK\carepd\CARE-PD*
