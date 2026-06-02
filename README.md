# Waymo 궤적 예측 — Maneuver-Conditioned Goal Trajectory Prediction

[Waymo Open Motion Dataset (WOMD) v1.3.0](https://waymo.com/open/data/motion/) 기반 자율주행 궤적 예측.  
기동 유형(Stop/Straight/LC/Turn) 조건화 + 목표점 예측으로 희귀 기동(Turn/LC)의 FDE를 크게 개선합니다.

---

## 모델 아키텍처

```
ego_hist      [B, 11, 10] ─┐
social_agents [B, 31, 11, 10] ─┤  Joint Transformer Encoder  →  global_feat [B, D]
map_scene     [B, 50, 10,  8] ─┤  (maneuver prefix token 포함, 99 tokens)  lane_feat [B, 50, D]
traffic       [B,  6,  1]  ─┘

                    ┌──────────────────────────────────────────┐
global_feat ───────►│  GoalHead (MLP)  →  goals [B, K, 2]     │  (use_goal_cond)
lane_feat   ───────►│  LaneGoalHead (cross-attn) → goals [B, K, 2] │  (use_lane_goal)
                    └──────────────────────────────────────────┘

[global_feat | mode_emb | goal_emb]  →  Causal Transformer Decoder  →  trajectory [B, K, 80, 2]
```

### 핵심 설계

| 구성요소 | 선택 | 설명 |
|---------|------|------|
| 인코더 | Joint Transformer | maneuver prefix token + ego/social/map/traf 99토큰 self-attention |
| 조건화 | Maneuver label (9-dim) | 6개 기동 + 3개 속도 원-핫, risk prefix로 인코더에 주입 |
| 디코더 | Causal Transformer | GPT-style, time-step 간 causal self-attn + lane cross-attn |
| Goal 헤드 | GoalHead / LaneGoalHead | 종점을 먼저 예측 → 디코더가 goal을 향해 궤적 생성 |
| Loss | Laplace NLL + RareAlign | 희귀 기동(Turn/LC)에 alignment loss 가중치 부여 |

---

## 실험 결과

### 전체 성능 (val, 2361 시나리오)

| 모델 | Best EP | minADE (m) | minFDE (m) | MR |
|------|:-------:|:----------:|:----------:|:---:|
| Baseline (Causal + ManeuverCond + RareAlign) | 20 | 1.732 | 4.789 | 0.499 |
| + GoalCond (free-form MLP goal) | 37 | **1.466** | 4.012 | 0.457 |
| + LaneGoal (lane-anchored cross-attn goal) | 33 | 1.486 | **4.006** | **0.454** |

### Per-Maneuver minADE (m)

| 모델 | Stop | Straight | LC_Left | LC_Right | Turn_Left | Turn_Right |
|------|:----:|:--------:|:-------:|:--------:|:---------:|:----------:|
| Baseline | 0.051 | 1.771 | 2.310 | 2.310 | 4.189 | 4.891 |
| GoalCond | **0.037** | 1.597 | **1.805** | **2.114** | 3.586 | **3.107** |
| LaneGoal | 0.050 | **1.538** | 1.922 | 2.275 | **3.527** | 3.859 |

### Per-Maneuver minFDE (m)

| 모델 | Stop | Straight | LC_Left | LC_Right | Turn_Left | Turn_Right |
|------|:----:|:--------:|:-------:|:--------:|:---------:|:----------:|
| Baseline | 0.127 | 4.709 | 6.409 | 6.237 | 12.317 | 14.521 |
| GoalCond | **0.112** | 4.354 | **4.759** | **5.478** | 9.998 | **8.747** |
| LaneGoal | 0.124 | **4.137** | 5.100 | 5.549 | **9.863** | 10.620 |

> GoalCond: Turn_Right ADE 4.891→3.107 (-36.5%), Turn_Right FDE 14.521→8.747 (-39.8%)

평가 재현:
```bash
conda run -n waymo python scripts/eval_maneuver_breakdown.py
```

---

## 학습

```bash
# Baseline
conda run -n waymo python train.py \
  --run_name baseline \
  --use_cache --cond_type maneuver --no_lane_mamba \
  --lam_div 0.1 --lam_align 1.0 --rare_align \
  --use_causal_attn --epochs 40 --lr 1e-4 --wd 1e-4

# + GoalCond
conda run -n waymo python train.py \
  --run_name exp_goal_cond \
  --use_cache --cond_type maneuver --no_lane_mamba \
  --lam_div 0.1 --lam_align 1.0 --rare_align \
  --use_causal_attn --epochs 40 --lr 1e-4 --wd 1e-4 \
  --use_goal_cond --lam_goal 0.5

# + LaneGoal
conda run -n waymo python train.py \
  --run_name exp_lane_goal \
  --use_cache --cond_type maneuver --no_lane_mamba \
  --lam_div 0.1 --lam_align 1.0 --rare_align \
  --use_causal_attn --epochs 40 --lr 1e-4 --wd 1e-4 \
  --use_lane_goal --lam_lane_goal 0.5
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--use_causal_attn` | False | Causal Transformer 디코더 사용 |
| `--use_goal_cond` | False | MLP goal head 활성화 |
| `--use_lane_goal` | False | Lane cross-attn goal head 활성화 |
| `--lam_goal` | 0.5 | Goal regression loss 가중치 |
| `--lam_lane_goal` | 0.5 | Lane goal CE loss 가중치 |
| `--rare_align` | False | 희귀 기동 alignment loss |
| `--lam_align` | 1.0 | Alignment loss 가중치 |
| `--no_lane_mamba` | False | LaneMamba 비활성화 |
| `--cond_type` | `maneuver` | `maneuver` \| `risk` |
| `--epochs` | 40 | 에폭 수 |
| `--lr` | 1e-4 | 학습률 |

---

## 체크포인트

| 체크포인트 | minADE | 설명 |
|-----------|:------:|------|
| `checkpoints/causal_maneuver_rare_align/model_best.pt` | 1.732 | Baseline |
| `checkpoints/exp_goal_cond/model_best.pt` | **1.466** | GoalCond (최고 ADE) |
| `checkpoints/exp_lane_goal/model_best.pt` | 1.486 | LaneGoal (최고 FDE) |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **minADE** | K=6 모드 중 GT에 가장 가까운 모드의 전 스텝 평균 L2 거리 (m) |
| **minFDE** | K=6 모드 중 GT에 가장 가까운 모드의 마지막 스텝 L2 거리 (m) |
| **MR** | minFDE > 2.0 m 이면 1 (miss rate) |

`src/eval/metrics.py` 에서 계산.

---

## 데이터

WOMD v1.3.0

| 분할 | 사용 샤드 | 위치 |
|------|-----------|------|
| Train | 00000–00049 (50 / 1000) | `waymo-motion-v1_3_0/train/` |
| Val   | 00000–00007 (8 / 150)   | `waymo-motion-v1_3_0/val/`   |

전처리:
```bash
conda run -n waymo python preprocess.py
```

---

## 디렉토리 구조

```
waymo_traj/
├── train.py
├── preprocess.py
├── scripts/
│   ├── eval_maneuver_breakdown.py   # per-maneuver ADE/FDE/MR 테이블
│   ├── visualize_goal_cond.py       # Baseline vs GoalCond 시각화
│   └── viz_maneuver_ablation.py
└── src/
    ├── data/
    │   ├── tfrecord.py
    │   └── features.py
    ├── models/
    │   ├── encoders.py       # MultiStreamMambaEncoder (Joint Transformer)
    │   ├── motion_model.py   # RiskConditionedModel, GoalHead, LaneGoalHead
    │   └── baselines.py
    └── eval/
        └── metrics.py
```

---

## 설치

```bash
conda create -n waymo python=3.10
conda activate waymo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 라이선스

Waymo Open Dataset은 [이용약관](https://waymo.com/open/terms/)에 따라 사용합니다. 모델 코드는 MIT 라이선스입니다.
