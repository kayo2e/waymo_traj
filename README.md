# Waymo 궤적 예측 — Risk-Conditioned Trajectory Prediction

[Waymo Open Motion Dataset (WOMD) v1.3.0](https://waymo.com/open/data/motion/) 기반 자율주행 궤적 예측.
위험 이벤트(근접·급제동·차선변경)를 감지해 궤적 예측에 직접 조건화하는 것이 핵심입니다.

---

## 현재 모델 아키텍처 (RiskConditionedModel)

```
map_scene     [B, 50, 10, 3] ─┐
social_agents [B, 31, 11,  6] ─┤  Joint Transformer Encoder  →  global_feat [B, D]
ego_hist      [B, 11,      6] ─┤  (self-attention, 98 tokens)     lane_feat [B, 50, D]
traffic       [B,  6,      1] ─┘

global_feat → risk_head → risk_logits [B, 3]   ← BCE loss

                    ┌──────────────────────────────────┐
risk_logits         │   RiskAwareLaneMamba              │
lane_feat    ──────►│   (HF Mamba, lateral 정렬)        │──► lane_context [B, D]
map_scene           │   lane_prob [B, 50]  ← 로깅용     │
                    └──────────────────────────────────┘

global_feat + lane_context  →  RiskFusion  →  context [B, D*2]

context + K mode_queries  →  AR LSTMCell (T=80 steps)  →  trajectory [B, K, 80, 2]
```

### 주요 설계 결정

| 구성요소 | 선택 | 이유 |
|---------|------|------|
| 인코더 | Joint Transformer (self-attention) | 98 토큰에서 양방향 어텐션이 Mamba보다 유리. MTR 등 SOTA 모두 Transformer 기반 |
| 차선 스캔 | RiskAwareLaneMamba (HF Mamba) | lateral 정렬된 50차선 시퀀스에서 위험 조건부 순차 추정 — Mamba가 의미 있는 지점 |
| 디코더 | AR LSTMCell + scheduled sampling | delta_xy 누적으로 절대좌표 생성, tf_prob: 1.0→0.2 |
| Loss | Laplace GMM NLL | K개 모드 전부 gradient 수신 → mode collapse 방지 |
| Risk 조건화 | GT label 주입(학습) / 자체 예측(추론) | teacher forcing으로 risk-aware 표현 강제 학습 |

### 입력 텐서

| 텐서 | 형상 | 설명 |
|------|------|------|
| `ego_hist` | `[B, 11, 6]` | 에고 과거 1.1초 (x, y, vx, vy, cos_h, sin_h) |
| `social_agents` | `[B, 31, 11, 6]` | 주변 에이전트 시계열 |
| `map_scene` | `[B, 50, 10, 3]` | 차선별 10개 폴리라인 포인트 (ego-relative) |
| `traffic` | `[B, 6, 1]` | 신호등 상태 |
| `risk_label` | `[B, 3]` | Proximity / HardBrake / LaneChange 이진 레이블 |

### 파라미터 수

| 모드 | 파라미터 |
|------|---------|
| `use_lane_mamba=True` (기본) | ~11.4M |
| `use_lane_mamba=False` (ablation) | ~1.2M |

---

## 위험 이벤트 레이블

`extract_risk_label(scenario)` — 에고 차량의 T_HIST(1.1초) 구간에서 추출

| 인덱스 | 이벤트 | 판정 기준 |
|--------|--------|-----------|
| [0] | Proximity | 임의 에이전트와 거리 ≤ 7 m |
| [1] | HardBrake | 종방향 감속도 ≤ −6.0 m/s² |
| [2] | LaneChange | 횡방향 속도 > 1.0 m/s (근사) |

---

## 실험 계획 및 현황

### Baseline: MTR (Motion Transformer)

30 에폭 학습 완료 (2026-05-10). 비교 기준점.

```
                  mAP    minADE   minFDE  MissRate
Vehicle         0.2219   1.4260   2.6283    0.3265
Pedestrian      0.3070   0.5336   1.1274    0.1296
Cyclist         0.2034   1.0043   1.9877    0.3168
─────────────────────────────────────────────────
Avg             0.2441   0.9880   1.9145    0.2576
```

---

### Phase 1: Laplace GMM NLL — 처음부터 재학습 ✅ 구현 완료

**목표**: soft-WTA의 mode collapse 문제를 Laplace GMM NLL로 해결.

**변경 내용**:
- Loss: `log(K) - logsumexp_k(-L1_k / b)` — K개 모드 전부 GT에 대해 gradient 수신
- 처음부터 Laplace로 학습 (중간 전환은 LR이 이미 0이 되어 무효)
- `--loss laplace --laplace_b 1.0` (기본값)

**이전 실패 원인**: WTA로 10 에폭 후 Laplace로 전환 → CosineAnnealingLR T_max=10 기준이라 epoch 5부터 LR≈0, 전환 효과 없음.

**실행**:
```bash
cd waymo_traj
conda activate waymo
python train.py --epochs 50
```

---

### Phase 2: RiskAwareLaneMamba ✅ 구현 완료

**목표**: 위험 이벤트 조건 하에 어느 차선이 안전한지를 Mamba로 순차 추정.

**구조**:
```
lane_feat [B, 50, D]  (Joint Transformer 어텐션 후)
  → lateral y 기준 정렬 (좌 → 현재 → 우)
  → concat(risk_emb broadcast, lane_sorted)
  → HF MambaModel (1 layer)
  → lane_prob [B, 50]  softmax 안전 확률
  → weighted sum → lane_context [B, D]

decoder_input = global_feat + lane_context  →  RiskFusion  →  AR decoder
```

**Mamba를 차선 스캔에 쓰는 이유**: 전체 98 토큰 인코딩에는 Mamba가 불리하지만 (양방향 어텐션 불가), lateral 정렬된 50 차선 시퀀스 스캔은 순차적 구조가 의미 있어 Mamba 적합.

**ablation 플래그**: `--no_lane_mamba` 로 비활성화해 ablation 비교 가능.

**실행**:
```bash
# Phase 2 포함 (기본)
python train.py --epochs 50

# ablation: RiskAwareLaneMamba 없는 버전
python train.py --epochs 50 --no_lane_mamba
```

**로깅**: 에폭마다 risk 있는/없는 시나리오에서 `max(lane_prob)` 비교 출력.
- risk 있을 때 max_lane_prob이 높을수록 → 모델이 특정 차선을 decisive하게 선택하고 있음.

---

### Phase 3: GPT-2 + LoRA AR 디코더 — 예정

**목표**: AR LSTMCell 디코더를 GPT-2 + LoRA로 교체 실험.

**방식**: 
- GPT-2 small (117M) + LoRA (rank=8, target: c_attn)
- `t=0..79` 각 스텝에서 `[mode_k | scene_ctx | prev_steps]` → Δxy 생성
- LSTMCell과의 차이: full attention over all previous steps (vs. fixed hidden state)
- `--decoder gpt2_lora` / `lstm` 플래그로 선택 가능하게 구현 예정

**핵심 실험 질문**: GPT-2의 80-step full attention이 LSTMCell 대비 trajectory에서 유의미한 개선을 주는가?

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **minADE** | K개 모드 중 최선의, 전체 유효 스텝 평균 L2 거리 (m) |
| **minFDE** | K개 모드 중 최선의, 마지막 유효 스텝 L2 거리 (m) |
| **MR** | minFDE > 2.0 m 이면 1 (miss) |

`src/eval/metrics.py`에서 계산.

---

## 학습 옵션

```bash
python train.py [옵션]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 50 | 에폭 수 |
| `--lr` | 1e-4 | 학습률 |
| `--wd` | 1e-4 | AdamW weight decay |
| `--lam` | 0.5 | L_risk 가중치 λ |
| `--loss` | `laplace` | `laplace` \| `wta` |
| `--laplace_b` | 1.0 | Laplace 스케일 b |
| `--no_lane_mamba` | False | RiskAwareLaneMamba 비활성화 (ablation) |
| `--log_every` | 50 | N 시나리오마다 로그 출력 |
| `--resume` | None | 체크포인트 경로 |
| `--max_scenarios` | None | 에폭당 최대 시나리오 수 (빠른 테스트용) |

체크포인트: `checkpoints/model_epoch{N:02d}.pt`, 최고 val ADE: `checkpoints/model_best.pt`

---

## 디렉토리 구조

```
waymo_traj/
├── train.py                  # 학습 진입점
├── evaluate.py               # 모델 평가
├── preprocess.py             # TFRecord → cache/ 전처리
├── config.yaml               # 하이퍼파라미터
├── requirements.txt
├── scripts/                  # 분석·시각화 보조 스크립트
│   ├── run_experiments.sh    # 실험 일괄 실행
│   ├── smoke_test.py         # 더미 데이터 동작 확인
│   ├── quick_validate.py     # 빠른 검증
│   ├── train_lstm.py         # LSTM 베이스라인 학습
│   ├── debug_lstm.py         # LSTM 디버그
│   ├── plot_curves.py        # 학습 곡선 시각화
│   ├── visualize_comparison.py
│   ├── visualize_modes.py
│   ├── visualize_rcm_compare.py
│   ├── visualize_risk_ablation.py
│   ├── visualize_3model_compare.py
│   └── viz_ablation_e73b04f.py
├── outputs/
│   └── logs/                 # 학습 로그 (gitignore 제외)
├── notebooks/
│   └── pipeline.ipynb
└── src/
    ├── data/
    │   ├── tfrecord.py       # TFRecord 이터레이터 (TF 없음)
    │   └── features.py       # 피처 추출 + extract_risk_label
    ├── models/
    │   ├── encoders.py       # JointPolylineEncoder
    │   ├── lane_mamba.py     # RiskAwareLaneMamba (HF Mamba 기반)
    │   ├── risk_fusion.py    # RiskFusion (global_feat + risk_emb concat)
    │   ├── motion_model.py   # RiskConditionedModel
    │   └── baselines.py      # ConstantVelocityBaseline, LSTMBaseline
    ├── eval/
    │   └── metrics.py        # minADE, minFDE, MR
    └── viz/
        └── plot.py
```

> `cache/`, `checkpoints/`, `libs/` 는 `.gitignore` 로 제외되어 있습니다.
> `cache/` 는 `python preprocess.py` 로 재생성하세요.

---

## 데이터

WOMD v1.3.0

| 분할 | 사용 샤드 | 위치 |
|------|-----------|------|
| Train | 00000–00049 (50 / 1000) | `waymo-motion-v1_3_0/train/` |
| Val   | 00000–00007 (8 / 150)   | `waymo-motion-v1_3_0/val/`   |

<details>
<summary>gsutil 다운로드 명령어</summary>

```bash
DATA=waymo-motion-v1_3_0
BASE=gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario

# Train (50 샤드)
gsutil -m cp \
  $(for i in $(seq 0 49); do
      echo "$BASE/training/training.tfrecord-$(printf '%05d' $i)-of-01000"
    done) $DATA/train/

# Val (8 샤드)
gsutil -m cp \
  $(for i in $(seq 0 7); do
      echo "$BASE/validation/validation.tfrecord-$(printf '%05d' $i)-of-00150"
    done) $DATA/val/
```

</details>

---

## 설치

```bash
conda create -n waymo python=3.10
conda activate waymo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install waymo-open-dataset-tf-2-11-0
```

> **Protobuf 주의**: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`은 `train.py` 실행 시 자동 설정됩니다.

---

## 라이선스

Waymo Open Dataset은 [이용약관](https://waymo.com/open/terms/)에 따라 사용합니다. 모델 코드는 MIT 라이선스입니다.
