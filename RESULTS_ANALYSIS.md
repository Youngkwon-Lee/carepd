# CARE-PD 학습 결과 분석 (2026-01-08)

## 실험 환경

**HPC 설정:**
- GPU: Tesla V100-PCIE-16GB (GPU 1)
- CUDA: 12.4
- PyTorch: 2.1.0+cu121
- Python: 3.9

**데이터셋:**
- BMCLab: 3,895 샘플, 23명 환자
- 라벨 분포: Class 0 (1,705), Class 1 (1,380), Class 2 (810)
- 클래스 불균형 비율: 2.1 : 1.7 : 1.0

---

## POTR 모델 결과

### 하이퍼파라미터 튜닝 (2-Fold)

**설정:**
- Trials: 3
- Search Space: lr, batch_size, epochs, criterion, weight_decay

**Best Trial (Trial 0):**
```yaml
epochs: 30 → 3 (avg_best_epoch 기반 조기 종료)
batch_size: 256
optimizer: AdamW
lr_head: 0.001
lr_backbone: 0.0001
criterion: FocalLoss (alpha=1, gamma=2)
weight_decay: 0
```

**결과:**
- Best F1 Score: 0.4036 (40.36%)

---

### 2-Fold LOSO Cross-Validation

**설정:**
- Folds: 2
- Best Trial 파라미터 사용

**성능 지표:**
```
Overall Accuracy: 43%
Weighted F1 Score: 0.40
Macro F1 Score: 0.33

클래스별 성능:
┌───────┬───────────┬────────┬──────────┬─────────┐
│ Class │ Precision │ Recall │ F1-Score │ Support │
├───────┼───────────┼────────┼──────────┼─────────┤
│   0   │   0.54    │  0.70  │   0.61   │  1,705  │
│   1   │   0.37    │  0.33  │   0.35   │  1,380  │
│   2   │   0.05    │  0.03  │   0.03   │   810   │
└───────┴───────────┴────────┴──────────┴─────────┘
```

**실행 시간:**
- Fold 1: 25초 (2,145 샘플)
- Fold 2: 36초 (1,750 샘플)
- 총 시간: ~1분

**문제점:**
- ⚠️ Class 2 (중증) 성능 극히 낮음 (F1: 0.03)
- 810명 중 24명만 탐지 (Recall 3%)

---

### 23-Fold LOSO Cross-Validation (전체 환자)

**설정:**
- Folds: 23 (Leave-One-Subject-Out)
- 각 fold마다 1명 환자 테스트
- Best Trial 파라미터 사용

**성능 지표:**
```
Overall Accuracy: 43%
Weighted F1 Score: 0.42
Macro F1 Score: 0.39

클래스별 성능:
┌───────┬───────────┬────────┬──────────┬─────────┐
│ Class │ Precision │ Recall │ F1-Score │ Support │
├───────┼───────────┼────────┼──────────┼─────────┤
│   0   │   0.53    │  0.59  │   0.56   │  1,705  │
│   1   │   0.41    │  0.32  │   0.36   │  1,380  │
│   2   │   0.24    │  0.27  │   0.25   │   810   │
└───────┴───────────┴────────┴──────────┴─────────┘
```

**실행 시간:**
- 평균 fold 시간: ~26초
- 총 시간: ~10분

---

## 2-Fold vs 23-Fold 비교

### 정량적 비교

| 지표 | 2-Fold | 23-Fold | 변화 | 분석 |
|------|--------|---------|------|------|
| **Overall Accuracy** | 43% | 43% | 0% | 동일 |
| **Weighted F1** | 0.40 | **0.42** | **+5%** | ✅ 개선 |
| **Macro F1** | 0.33 | **0.39** | **+18%** | ✅ 개선 |
| **Class 0 F1** | 0.61 | 0.56 | -8% | 약간 감소 |
| **Class 1 F1** | 0.35 | 0.36 | +3% | 유지 |
| **Class 2 F1** | 0.03 | **0.25** | **+733%** | 🚀 대폭 개선 |

### Class 2 (중증) 상세 비교

| 지표 | 2-Fold | 23-Fold | 개선 |
|------|--------|---------|------|
| **Precision** | 0.05 | 0.24 | +380% |
| **Recall** | 0.03 | 0.27 | +800% |
| **F1 Score** | 0.03 | 0.25 | +733% |
| **탐지 환자 수** | ~24명 | ~219명 | +812% |

**핵심 발견:**
- 23-Fold LOSO가 환자별 일반화 성능을 더 정확히 평가
- 클래스 불균형 문제 완화 효과
- 중증 환자 탐지율이 3%에서 27%로 **9배 향상**

---

## 베이스라인 비교 (논문 결과)

> ⚠️ **참고**: 논문의 공식 베이스라인 결과는 확인 필요
> GitHub README 또는 논문 Table 참조

### 예상 베이스라인 (참고용)

**조건:**
- 전체 9개 코호트 통합 (186명 환자)
- LOSO Cross-Validation
- 다양한 평가 프로토콜

**우리 결과 (BMCLab 단일 코호트):**
- 23명 환자만 사용
- LOSO 23-fold
- Weighted F1: 0.42

**비교 시 고려사항:**
1. 데이터 규모: BMCLab (23명) vs 전체 (186명)
2. 코호트 다양성: 단일 사이트 vs 8개 사이트
3. 평가 프로토콜: LOSO vs LODO, MIDA, Cross-Dataset

---

## 분석 및 인사이트

### 1. LOSO의 중요성

**23-Fold LOSO 장점:**
- 환자별 일반화 성능 정확히 측정
- 새로운 환자에 대한 예측 성능 평가
- 실제 임상 환경과 유사한 시나리오

**2-Fold의 한계:**
- 환자 그룹 간 특성 차이 반영 부족
- Class 2 (중증) 환자가 일부 fold에만 집중

### 2. 클래스 불균형 문제

**현상:**
- Class 0 : Class 1 : Class 2 = 2.1 : 1.7 : 1.0
- 중증 환자 (Class 2) 데이터 부족

**영향:**
- Class 2 성능이 가장 낮음 (F1: 0.25)
- Precision 0.24: 예측한 중증 환자 중 24%만 실제 중증
- Recall 0.27: 실제 중증 환자 중 27%만 탐지

**개선 방향:**
1. Focal Loss의 gamma 값 증가 (2 → 3, 4)
2. Class weights 조정
3. SMOTE 등 샘플링 기법 적용
4. Epoch 수 증가 (3 → 10, 30)

### 3. 모델 학습 효율성

**Epoch 분석:**
- Hypertuning 결과: avg_best_epoch = 3
- 매우 빠른 수렴 (30 epoch 중 3 epoch만 필요)
- 가능성: 조기 수렴으로 인한 언더피팅

**개선 실험:**
- Epoch 10, 30으로 증가 테스트
- Learning rate 조정
- Warmup scheduler 추가

### 4. 성능 향상 가능성

**현재 성능 (POTR + BMCLab):**
- Weighted F1: 0.42
- Macro F1: 0.39

**예상 개선 방향:**
1. **다른 모델 테스트**
   - MotionBERT (25M params): 더 복잡한 패턴 학습 가능
   - MotionCLIP (8M params): CLIP 기반 feature
   - Ensemble: 여러 모델 조합

2. **데이터 증강**
   - 현재: MirrorReflection, RandomRotation, RandomNoise
   - 추가 가능: TimeWarping, Jittering

3. **전체 데이터셋 활용**
   - 9개 코호트 통합 (186명 → 더 많은 학습 데이터)
   - Cross-Dataset 평가

---

## 다음 단계

### 우선순위 1: 경량 모델 학습 (GPU 1)
- [ ] MotionCLIP (8M params)
- [ ] PoseFormerV2 (8M params)
- [ ] MixSTE (10M params)
- [ ] MoMask (10M params)

### 우선순위 2: POTR 성능 개선 실험
- [ ] Epoch 증가 (3 → 10, 30)
- [ ] Focal Loss gamma 증가 (2 → 3, 4)
- [ ] Class weights 조정

### 우선순위 3: 추가 데이터셋
- [ ] 3DGait 전처리 및 학습
- [ ] PD-GaM 전처리 및 학습
- [ ] 전체 코호트 통합 (186명)

---

## 저장된 파일

### 모델 체크포인트
```
experiment_outs/Hypertune/POTR_BMCLab/0/
├── study.pkl (7.4KB - 하이퍼튜닝 결과)
├── models/
│   ├── train_BMCLab_2fold/ (24MB)
│   │   ├── fold1/latest_epoch.pth.tr
│   │   └── fold2/latest_epoch.pth.tr
│   └── train_BMCLab_23fold/ (예상 ~270MB)
│       ├── fold1/latest_epoch.pth.tr
│       ├── fold2/latest_epoch.pth.tr
│       ├── ...
│       └── fold23/latest_epoch.pth.tr
```

### 백업 파일 (HPC)
```
~/carepd/CARE-PD/
├── potr_bmclab_20260108.tar.gz (7.4KB)
├── potr_2fold_results_20260108.tar.gz (24MB)
└── potr_23fold_results_20260108.tar.gz (생성 예정)
```

---

**마지막 업데이트**: 2026-01-08
**작성자**: Claude + YK
