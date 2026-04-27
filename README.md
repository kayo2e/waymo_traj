# Waymo 궤적 예측 — 위험 조건부 Mamba 모델

[Waymo Open Motion Dataset (WOMD) v1.3.0](https://waymo.com/open/data/motion/) 기반 자율주행 궤적 예측.
위험 이벤트(근접·급제동·차선변경)를 감지해 궤적 예측에 직접 조건화하는 것이 핵심입니다.

## 전체 파이프라인

```
Stage 1  씬 파싱 & 위험 이벤트 레이블링
           Proximity / HardBrake / LaneChange → risk_label [B, 3]
  ↓
Stage 2  MultiStreamMambaEncoder
           Temporal / Traffic / Scene 3개 독립 스트림 → GlobalMamba → global_feat [B, D]
           risk_head → risk_logits [B, 3]   (위험 분류)
           RiskFusion → fused [B, D*2]      (위험 조건화)
  ↓
Stage 3  K=6 다중 궤적 가설 디코딩
           KeypointDecoder  → keypoints  [B, K, 3, 2]   (1s/3s/5s)
           TrajectoryRefiner → trajectory [B, K, 80, 2]  (8초 @ 10Hz)
           + Gemini 자연어 설명 & 정제
```

**주요 설계 결정**

- **위험 조건화** — GT risk_label을 학습 시 직접 주입, 추론 시 모델이 자체 예측한 확률로 대체
- **3-스트림 Mamba** — Temporal / Traffic / Scene 독립 SSM → GlobalMamba 통합
- **Multi-task Loss** — `L_traj (WTA) + λ * L_risk (BCE)`, 기본 λ=0.5
- **TF 없는 파싱** — TFRecord를 `struct.unpack`으로 직접 읽어 protobuf 버전 충돌 차단

---

## 모델 구조 및 학습 원리

### 입력 형식

| 텐서 | 형상 | 설명 |
|------|------|------|
| `ego_hist` | `[B, 11, 6]` | 에고 과거 궤적 (x, y, vx, vy, cos_h, sin_h) |
| `social_agents` | `[B, 31, 6]` | 주변 에이전트 (T_HIST 평균) |
| `map_tokens` | `[B, 56, 3]` | 맵 폴리라인 50개 + 신호 6개 (max-pool → 3D) |
| `risk_label` | `[B, 3]` | Proximity / HardBrake / LaneChange 이진 레이블 |

### 전체 구조도

```
ego_hist      [B, 11, 6]  → TemporalMamba  → mean → [B, D]  ─┐
social_agents [B, 31, 6]  → TrafficMamba   → mean → [B, D]  ─┤ stack [B, 3, D]
map_tokens    [B, 56, 3]  → SceneMamba     → mean → [B, D]  ─┘
                                                                ↓
                                                          GlobalMamba
                                                                ↓
                                                     global_feat [B, D]
                                                                ↓
                          risk_head (Linear) → risk_logits [B, 3]  ← BCE loss
                                                                ↓
risk_label [B, 3] ──────── RiskFusion (Linear+ReLU+Linear) ──→ fused [B, D*2]
                                                                ↓
                                              × K mode_queries [K, D]
                                                                ↓
                                                    combined [B, K, D*3]
                                                                ↓
                         KeypointDecoder (MLP) → keypoints  [B, K, 3, 2]
                         TrajectoryRefiner (MLP) → trajectory [B, K, 80, 2]
```

### 인코딩 단계 — MultiStreamMambaEncoder

각 모달리티를 **독립된 Mamba SSM**으로 처리합니다. Transformer의 attention과 달리 선택적 상태 방정식(SSM)으로 순환 처리하므로 시퀀스 길이에 대해 O(N)입니다.

```python
e = Linear(agent_dim → D)(ego_hist)        # [B, 11, D]
s = Linear(agent_dim → D)(social_agents)   # [B, 31, D]
m = Linear(map_dim   → D)(map_tokens)      # [B, 56, D]

e_feat = TemporalMamba(e).mean(dim=1)      # [B, D]
s_feat = TrafficMamba(s).mean(dim=1)       # [B, D]
m_feat = SceneMamba(m).mean(dim=1)         # [B, D]

combined   = stack([e_feat, s_feat, m_feat], dim=1)  # [B, 3, D]
global_out = GlobalMamba(combined)                    # [B, 3, D]
global_feat = global_out.mean(dim=1)                  # [B, D]
```

### 위험 조건화 — RiskFusion

```python
risk_logits = Linear(D → 3)(global_feat)         # [B, 3]  ← 위험 분류

# 학습 시: GT 이진 레이블 사용 (teacher forcing)
# 추론 시: sigmoid(risk_logits).detach() 사용
risk_emb = Sequential(Linear(3→D), ReLU, Linear(D→D))(risk_input)  # [B, D]
fused    = cat([global_feat, risk_emb], dim=-1)                     # [B, D*2]
```

### 디코딩 단계 — K=6 다중 가설

```python
combined = cat([fused × K, mode_queries], dim=-1)  # [B, K, D*3]

kp   = MLP(D*3 → D → 6)(combined).reshape(B, K, 3, 2)    # 웨이포인트
traj = MLP(D*3+6 → 512 → 256 → 160)(cat([combined, kp_flat]))
           .reshape(B, K, 80, 2)                           # 전체 궤적
```

### 학습 방식 — Multi-task Loss + WTA

```python
# Winner-Takes-All: K개 중 GT에 가장 가까운 mode만 역전파
best_k = argmin([mean_L2(traj[k], gt) for k in range(K)])

L_traj = huber(kp[best_k],   gt_kp)   +  l1(traj[best_k], gt_traj)
L_risk = BCEWithLogitsLoss(risk_logits, risk_label_gt)

loss   = L_traj + λ * L_risk    # λ 기본값 0.5
```

### 구성 요소 요약

| 구성요소 | 종류 | 역할 |
|---------|------|------|
| TemporalMamba | Mamba SSM | 에고 시계열 처리 |
| TrafficMamba | Mamba SSM | 주변 에이전트 상호작용 |
| SceneMamba | Mamba SSM | 맵 + 신호 처리 |
| GlobalMamba | Mamba SSM | 3 스트림 통합 |
| risk_head | Linear | 위험 이벤트 분류 (→ BCE loss) |
| RiskFusion | MLP | 위험 레이블 조건화 |
| mode_queries | Embedding | K개 행동 의도 학습 |
| kp_head / refiner | MLP | 웨이포인트 & 궤적 디코딩 |

**총 파라미터: 41M**

---

## 위험 이벤트 레이블

`extract_risk_label(scenario)` — 에고 차량의 T_HIST(1.1초) 구간에서 추출

| 인덱스 | 이벤트 | 판정 기준 |
|--------|--------|-----------|
| [0] | Proximity | 임의 에이전트와 거리 ≤ 7 m |
| [1] | HardBrake | 종방향 감속도 ≤ −6.0 m/s² |
| [2] | LaneChange | 횡방향 속도 > 1.0 m/s (근사) |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **minADE** | K개 mode 중 최선의, 전체 유효 스텝 평균 L2 거리 (m) |
| **minFDE** | K개 mode 중 최선의, 마지막 유효 스텝 L2 거리 (m) |
| **MR** | minFDE > 2.0 m 이면 1 (miss), 아니면 0 |

모두 `src/eval/metrics.py`에서 계산합니다.

---

## 디렉토리 구조

```
waymo_traj/
├── train.py                  # 학습 진입점 (RiskConditionedModel)
├── evaluate.py               # 베이스라인 비교 테이블
├── smoke_test.py             # 더미 데이터 동작 확인
├── config.yaml               # 하이퍼파라미터
├── requirements.txt
├── notebooks/
│   └── pipeline.ipynb        # 엔드투엔드 데모 노트북
└── src/
    ├── data/
    │   ├── tfrecord.py       # TFRecord 이터레이터 (TF 없음)
    │   └── features.py       # 피처 추출 + extract_risk_label
    ├── models/
    │   ├── encoders.py       # TambaMambaEncoder, JointPolylineEncoder,
    │   │                     # MultiStreamMambaEncoder
    │   ├── risk_fusion.py    # RiskFusion
    │   ├── motion_model.py   # RiskConditionedModel, WaymoMotionModel
    │   └── baselines.py      # ConstantVelocityBaseline, LSTMBaseline
    ├── pipeline/
    │   ├── stage1.py         # 위험 이벤트 감지
    │   ├── stage2.py         # Mamba 인코딩 브릿지
    │   └── stage3_gemini.py  # Gemini 설명 & 정제
    ├── eval/
    │   └── metrics.py        # minADE, minFDE, MR
    └── viz/
        └── plot.py           # 궤적 시각화
```

---

## 데이터

WOMD v1.3.0 — `gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/`에서 다운로드

| 분할 | 사용 샤드 | 위치 |
|------|-----------|------|
| Train | 00000–00049 (50 / 1000) | `waymo-motion-v1_3_0/train/` |
| Val   | 00000–00007 (8 / 150)   | `waymo-motion-v1_3_0/val/`   |
| Test  | 00000–00007 (8 / 150)   | `waymo-motion-v1_3_0/test/`  |

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

# Test (8 샤드)
gsutil -m cp \
  $(for i in $(seq 0 7); do
      echo "$BASE/testing/testing.tfrecord-$(printf '%05d' $i)-of-00150"
    done) $DATA/test/
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

> **Protobuf 주의**: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`은 `train.py`와 `evaluate.py` 실행 시 자동으로 설정됩니다.

---

## 사용법

### 동작 확인 (smoke test)

```bash
cd waymo_traj
python smoke_test.py
```

더미 데이터로 forward / backward / 지표 계산이 모두 통과하는지 확인합니다.

### 학습

```bash
python train.py [--epochs 50] [--lr 1e-4] [--wd 1e-4] [--lam 0.5] [--device cuda]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 50 | 에폭 수 |
| `--lr` | 1e-4 | 학습률 |
| `--wd` | 1e-4 | AdamW weight decay |
| `--lam` | 0.5 | L_risk 가중치 λ |
| `--log_every` | 50 | N 시나리오마다 로그 출력 |
| `--resume` | None | 이어서 학습할 체크포인트 경로 |

체크포인트는 `checkpoints/model_epoch{N:02d}.pt`, val minADE 최고는 `checkpoints/model_best.pt`에 저장됩니다.

### 평가

```bash
python evaluate.py \
    --ckpt      checkpoints/model_best.pt \
    --lstm_ckpt checkpoints/lstm_best.pt \
    --device cuda [--max_scenarios 200]
```

3개 모델의 minADE / minFDE / MR 비교 테이블을 출력합니다.

#### 실험 결과 (quick-test: val 200 시나리오, 3 epoch)

```
==================================================================
  Baseline Comparison vs WOMD Ground Truth  (val set)
==================================================================
  모델                          minADE (m)  minFDE (m)   MR (2m)
  --------------------------  ----------  ----------   -------
  Constant Velocity                 5.504      15.675     0.670
  LSTM (trained)                    5.586      15.384     0.710
  RiskConditionedModel K=6          3.439       8.975     0.570
==================================================================
```

| 항목 | 설명 |
|------|------|
| **조건** | val 200 시나리오, 3 epoch (quick-test) / GPU: CUDA |
| **CV 대비 개선** | minADE **-37.5%** / minFDE **-42.7%** / MR **-15%** |
| **LSTM 결과** | 300 시나리오·3 에폭 underfitting → CV와 유사 수준. 충분히 학습 시 4-5m 수준 |
| **WOMD SOTA** | ~0.3-0.5 m (50 에폭 + 전체 1000 shard 기준) |

> K=6 다중 모드 예측(WTA), MultiStreamMamba 인코더, 위험 컨디셔닝 조합이 단순 baselines 대비 유의미한 성능 개선을 보입니다.

### 노트북

`notebooks/pipeline.ipynb`를 `waymo_kernel` 커널로 실행하면 Stage 1~3 전체와 Gemini 설명, 궤적 시각화를 대화형으로 확인할 수 있습니다.

---

## Stage 3: Gemini 연동

`src/pipeline/stage3_gemini.py`가 제공하는 함수:

| 함수 | 설명 |
|------|------|
| `generate_gemini_explanation` | 위험 레이블 + 예측 웨이포인트 → `[주체]+[행동]+[제약]` 안전 분석 텍스트 |
| `mamba_context_to_text` | Mamba attention norm → 자연어 요약 |
| `gemini_refine_trajectory` | 물리 일관성 기반 80스텝 궤적 보정 |

모델 우선순위: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.0-flash-lite`

사용 전 `GOOGLE_API_KEY` 환경변수를 설정하세요.

---

## 라이선스

Waymo Open Dataset은 [이용약관](https://waymo.com/open/terms/)에 따라 사용합니다. 모델 코드는 MIT 라이선스입니다.
