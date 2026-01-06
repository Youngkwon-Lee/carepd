# CARE-PD Model Architectures Analysis

> 7개 모션 인코더 백본 모델 상세 분석

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Motion Encoder Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│  Input: 3D Pose Sequence [B, T, J, C]                               │
│         B=Batch, T=Frames, J=Joints, C=Channels                      │
│                              ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Backbone Model                              │  │
│  │  (POTR / MotionBERT / MoMask / MotionCLIP / etc.)            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  Feature: [B, T, J, dim_rep] or [B, dim_rep]                        │
│                              ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Classifier Head                              │  │
│  │  FC → BatchNorm → ReLU → Dropout → FC → Softmax               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  Output: UPDRS Score Prediction [B, num_classes]                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. MotionBERT (DSTformer)

### 개요
- **유형**: 2D→3D Lifting Model
- **입력**: 2D poses [B, T, J, 2]
- **출력**: [B, T, J, dim_rep=512]

### 핵심 아키텍처
```
DSTformer (Dual-Stream Spatio-Temporal Transformer)
├── joints_embed: Linear(3 → dim_feat=256)
├── pos_embed: [1, J, dim_feat] (학습 가능)
├── temp_embed: [1, T_max, 1, dim_feat] (학습 가능)
├── blocks_st: [depth] × Block(stage_st)  # Spatial → Temporal
├── blocks_ts: [depth] × Block(stage_ts)  # Temporal → Spatial
├── Adaptive Fusion (α-weighted sum)
├── pre_logits: Linear(dim_feat → dim_rep) + Tanh
└── head: Linear(dim_rep → dim_out)
```

### 핵심 특징
- **Dual-Stream**: ST 브랜치와 TS 브랜치 병렬 처리
- **Adaptive Fusion**: 학습 가능한 가중치로 두 스트림 결합
- **Attention Modes**: spatial, temporal, series, parallel, coupling

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| dim_feat | 256 | Feature dimension |
| dim_rep | 512 | Representation dimension |
| depth | 5 | Transformer depth |
| num_heads | 8 | Attention heads |
| maxlen | 243 | Max sequence length |

---

## 2. POTR (Pose Transformer)

### 개요
- **유형**: 3D Single-View Model
- **입력**: 3D joints [B, T, pose_dim]
- **출력**: [T, B, model_dim]

### 핵심 아키텍처
```
PoseTransformer
├── pose_embedding: (Optional) Pose encoder
├── pos_encoder: PositionEncodings1D (sinusoidal)
├── transformer: Transformer Encoder
│   ├── num_encoder_layers: 6
│   ├── model_dim: 256
│   ├── num_heads: 8
│   └── dim_ffn: 2048
└── Output: Encoder memory states
```

### 핵심 특징
- **Non-Autoregressive**: 전체 시퀀스 동시 처리
- **Sinusoidal Position Encoding**: 고정된 위치 인코딩
- **Flexible Pose Embedding**: 다양한 pose encoder 지원

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| pose_dim | 54 | Pose dimension |
| model_dim | 256 | Model dimension |
| num_encoder_layers | 6 | Encoder layers |
| num_heads | 8 | Attention heads |
| dim_ffn | 2048 | FFN dimension |

---

## 3. MoMask (RVQVAE)

### 개요
- **유형**: VQ-VAE 기반 모션 모델
- **입력**: HumanML3D format [B, T, 263]
- **출력**: [B, code_dim=512, T/4]

### 핵심 아키텍처
```
RVQVAE (Residual Vector Quantized VAE)
├── encoder: Encoder (1D Conv + Dilated Conv)
│   ├── input_width: 263
│   ├── down_t: 3 (temporal downsampling)
│   ├── stride_t: 2
│   └── dilation_growth_rate: 3
├── quantizer: ResidualVQ
│   ├── num_quantizers: N
│   ├── nb_code: 1024 (codebook size)
│   └── code_dim: 512
└── decoder: Decoder (1D Conv + Upsampling)
```

### 핵심 특징
- **Residual VQ**: 다단계 양자화로 정밀한 표현
- **Temporal Downsampling**: 4배 다운샘플링 (T → T/4)
- **Dilated Convolutions**: 넓은 수용 영역

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| input_width | 263 | HumanML3D feature dim |
| nb_code | 1024 | Codebook size |
| code_dim | 512 | Code dimension |
| down_t | 3 | Downsampling factor |

---

## 4. MotionCLIP

### 개요
- **유형**: CLIP 기반 모션 표현 학습
- **입력**: 6D rotation [B, T, J, 6]
- **출력**: [B, latent_dim=512]

### 핵심 아키텍처
```
Encoder_TRANSFORMER (CVAE style)
├── Input Processing:
│   ├── njoints: 25
│   ├── nfeats: 6 (6D rotation)
│   └── num_frames: 60
├── Transformer Encoder:
│   ├── latent_dim: 512
│   ├── ff_size: 1024
│   └── num_layers: 8
└── Global rotation: [π, 0, 0]
```

### 핵심 특징
- **CLIP Alignment**: 텍스트-모션 정렬 학습
- **6D Rotation**: 연속적인 회전 표현
- **Global Pooling**: 시퀀스를 단일 벡터로 압축

---

## 5. MotionAGFormer

### 개요
- **유형**: Attention + Graph 하이브리드
- **입력**: 2D poses [B, T, J, 2]
- **출력**: [B, T, J, dim_rep=512]

### 핵심 아키텍처
```
MotionAGFormer
├── joints_embed: Linear(dim_in → dim_feat)
├── pos_embed: [1, J, dim_feat]
├── layers: [n_layers] × MotionAGFormerBlock
│   ├── Attention Branch (ST):
│   │   ├── att_spatial: AGFormerBlock(attention, spatial)
│   │   └── att_temporal: AGFormerBlock(attention, temporal)
│   ├── Graph Branch (ST):
│   │   ├── graph_spatial: AGFormerBlock(GCN, spatial)
│   │   └── graph_temporal: AGFormerBlock(GCN/MS-TCN, temporal)
│   └── Adaptive Fusion: α × Attention + (1-α) × Graph
├── norm: LayerNorm
└── rep_logit: Linear + Tanh
```

### 핵심 특징
- **Dual Branch**: Attention + Graph 병렬 처리
- **Adaptive Fusion**: 학습 가능한 브랜치 가중치
- **Multi-Scale TCN**: 시간축 다중 스케일 처리
- **Hierarchical Mode**: 계층적 특징 분리

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| n_layers | - | Number of blocks |
| dim_feat | - | Feature dimension |
| dim_rep | 512 | Representation dim |
| num_heads | 4 | Attention heads |
| neighbour_num | 4 | GCN neighbors |

---

## 6. MixSTE

### 개요
- **유형**: Mixed Spatio-Temporal Encoder
- **입력**: 2D poses [B, T, J, 2]
- **출력**: [B, T, J, embed_dim_ratio]

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| num_frame | T | Input frames |
| num_joints | 17 | Number of joints |
| embed_dim_ratio | - | Embedding ratio |
| depth | - | Transformer depth |

---

## 7. PoseFormerV2

### 개요
- **유형**: Pose Transformer V2
- **입력**: 2D poses [B, T, J, 2]
- **출력**: [B, 1, embed_dim_ratio × J × 2]

### 핵심 특징
- **Frequency Selection**: DCT 기반 주파수 선택
- **Frame Keeping**: 중요 프레임 선별

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| number_of_kept_frames | - | Kept frames |
| number_of_kept_coeffs | - | DCT coefficients |
| embed_dim_ratio | - | Embedding ratio |

---

## Classifier Head

### 구조
```python
ClassifierHead
├── Input Processing (backbone-specific):
│   ├── MotionBERT: [B, T, J, C] → temporal avg → [B, J, C] → optional merge → [B, C]
│   ├── POTR: [T, B, C] → [B, T, C] → temporal avg → [B, C]
│   ├── MoMask: [B, C, T] → temporal avg → [B, C]
│   └── MotionCLIP: [B, C] (already pooled)
├── FC Layers:
│   ├── fc1: Linear(input_dim, hidden_dim) + BatchNorm + ReLU
│   └── fc2: Linear(hidden_dim, num_classes)
└── Dropout: p=classifier_dropout
```

### Input Dimension 계산
| Backbone | merge_joints=True | merge_joints=False |
|----------|-------------------|---------------------|
| MotionBERT | dim_rep | dim_rep × num_joints |
| POTR | model_dim | model_dim × seq_len |
| MoMask | dim_rep | dim_rep × (T/4) |
| MotionCLIP | dim_rep | dim_rep |
| MotionAGFormer | dim_rep | dim_rep × num_joints |

---

## 학습 파이프라인

### End-to-End Training
```python
MotionEncoder(backbone, params)
├── backbone: Pretrained backbone (frozen or trainable)
├── head: ClassifierHead(num_classes=4)
└── forward(x, metadata, med, valid_mask):
    ├── feat = backbone(x)
    ├── feat += medication_embedding (optional)
    ├── feat += metadata_embedding (optional)
    └── return head(feat, valid_mask)
```

### Training Modes
1. **end2end**: 전체 모델 학습
2. **classifier_only**: backbone 고정, classifier만 학습

---

## PhysioKorea 통합 포인트

### 1. 모델 선택 가이드
| 사용 사례 | 추천 모델 | 이유 |
|----------|----------|------|
| 실시간 분석 | POTR | 경량, 빠른 추론 |
| 높은 정확도 | MotionAGFormer | Attention+Graph 하이브리드 |
| 영상 입력 | MotionBERT | 2D→3D lifting |
| 텍스트 연동 | MotionCLIP | 자연어 설명 가능 |

### 2. 코드 재사용
```python
# PhysioKorea 통합 예시
from model.backbone_loader import load_pretrained_backbone
from model.motion_encoder import MotionEncoder

params = {...}  # 설정 로드
backbone = load_pretrained_backbone(params, 'motionbert')
model = MotionEncoder(backbone, params, num_classes=4)
```

---

*분석일: 2025-01-06*
*CARE-PD Repository: C:\Users\YK\carepd\CARE-PD*
