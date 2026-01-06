# CARE-PD Dataset Reference

> **A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment**
>
> NeurIPS 2025 Datasets & Benchmarks Track

## Overview

| 항목 | 내용 |
|------|------|
| **GitHub** | https://github.com/TaatiTeam/CARE-PD |
| **arXiv** | https://arxiv.org/abs/2510.04312 |
| **웹사이트** | https://neurips2025.care-pd.ca |
| **HuggingFace** | https://huggingface.co/datasets/vida-adl/CARE-PD |
| **Dataverse** | https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/TWIKMK |
| **라이센스** | MIT License (코드) / CC BY-NC 4.0 (데이터) |

## Dataset Scale

- **8개 임상 사이트**에서 수집
- **9개 코호트** 통합
- **363명 파킨슨병 환자** (다양한 질병 심각도)
- 최초의 다중 사이트 3D 메시 보행 데이터 아카이브

---

## 9개 Cohort 상세

| Dataset | 설명 | 특징 |
|---------|------|------|
| **3DGait** | 임상 보행 기록 | UPDRS 점수 포함 |
| **BMCLab** | 보행 기록 | 약물 상태 + UPDRS 점수 |
| **DNE** | 다양한 신경학적 상태 | 건강인, PD, 기타 신경질환 |
| **E-LC** | 동결 이벤트 포함 | 약물 on/off 상태 |
| **KUL-DT-T** | Freezer 분류 | Freezer/Non-freezer |
| **PD-GaM** | 임상 보행 기록 | UPDRS 점수 포함 |
| **T-SDU** | 일반 보행 기록 | 건강인 데이터 |
| **T-SDU-PD** | PD 환자 보행 | UPDRS 점수 포함 |
| **T-LTC** | 추가 보행 기록 | 보조 데이터셋 |

---

## Data Structure

### 원본 형식 (SMPL)
```python
{
    "anonymized_subject_id": {
        "anonymized_walk_id": {
            "pose": array,      # SMPL pose parameters
            "trans": array,     # Translation data
            "beta": array,      # Body shape (zeros for privacy)
            "fps": int,         # Frames per second
            "UPDRS_GAIT": int,  # Clinical score (0-3) or None
            "medication": str,  # 약물 상태 or None
            "other": str        # 추가 레이블 or None
        }
    }
}
```

### 전처리 형식
```
assets/datasets/
├── 6D_SMPL/           # 6D 회전 표현
├── h36m/              # Human3.6M 형식
├── HumanML3D/         # HumanML3D 형식
├── folds/             # 교차검증 분할
└── *.pkl              # 원본 데이터셋 파일
```

---

## Implemented Models

### 3D Single-View Models
| 모델 | 입력 형식 | 설명 |
|------|----------|------|
| **POTR** | 3D joints | Pose Transformer |
| **MoMask** | HumanML3D | VQ-VAE 기반 모션 모델 |
| **MotionCLIP** | HumanML3D | CLIP 기반 표현 학습 |

### 2D-3D Multi-View Models
| 모델 | 입력 형식 | 설명 |
|------|----------|------|
| **MotionBERT** | 2D→3D | DSTformer 아키텍처 |
| **MixSTE** | 2D→3D | Spatio-Temporal Encoder |
| **PoseFormerV2** | 2D→3D | Pose Transformer V2 |
| **MotionAGFormer** | 2D→3D | Attention-Graph Transformer |

---

## Evaluation Protocols

| 프로토콜 | 설명 |
|----------|------|
| **LOSO** | Leave-One-Subject-Out |
| **Cross-Dataset** | 교차 데이터셋 일반화 |
| **LODO** | Leave-One-Dataset-Out |
| **MIDA** | Multi-dataset In-domain Adaptation |

---

## Technical Requirements

```bash
Python 3.9.21
PyTorch 2.6.0 + CUDA 11.8
```

### 주요 Dependencies
- einops, optuna, smplx, timm, torch_dct, trimesh, wandb, scikit_learn

---

## Quick Start

```bash
# 데이터셋 다운로드
huggingface-cli download vida-adl/CARE-PD --repo-type dataset --local-dir ./assets/datasets

# 모델 다운로드
bash scripts/download_models.sh

# 학습/평가
bash scripts/eval_within_dataset.sh
```

---

## Citation

```bibtex
@inproceedings{adeli2025carepd,
  title={CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment},
  author={Vida Adeli et al.},
  booktitle={NeurIPS},
  year={2025}
}
```

---

*문서 작성일: 2025-01-06*
*저장소: C:\Users\YK\carepd\CARE-PD*
