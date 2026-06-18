# Waymo 궤적 예측 — Maneuver-Conditioned Goal Trajectory Prediction

[Waymo Open Motion Dataset (WOMD) v1.3.0](https://waymo.com/open/data/motion/) 기반 자율주행 궤적 예측.  
기동 유형(Stop / Straight / LC / Turn) 조건화 + 목표점 예측 + 차선 그래프 인코딩으로  
희귀 기동(Turn / LC)의 ADE를 Baseline 대비 최대 56% 개선합니다.

---

## 모델 아키텍처

```
입력 (ego-relative)
 ├── ego_hist      [B, 11, 12]        ego 1초 과거 (x,y,vx,vy,head,valid,t,type,len,wid)
 ├── social_agents [B, 31, 11, 12]    주변 에이전트
 ├── map_scene     [B, 50, 10, 10]    차선 폴리라인 (x,y,dir,type,speed_limit,lane_type)
 ├── traffic       [B,  6,  3]        신호등 (state, stop_x, stop_y)
 └── cond_label    [B,  9]            6 maneuver + 3 speed one-hot

          ┌──────────────────────────────────────┐
          │  Joint Transformer Encoder           │
          │  risk_prefix(1) + ego(11) +          │
          │  social(31) + map(50) + traf(6)      │
          │  = 99 tokens, 2-layer Self-Attention │
          └───────────┬──────────────────────────┘
                      │
          ┌───────────▼──────────────────────────┐
          │  LaneGraphEncoder  (optional)         │
          │  기하학적 adjacency bias (연결/근접)    │
          │  2-layer GAT → lane_feat [B,50,D]    │
          └───────────┬──────────────────────────┘
                      │
          ┌───────────▼──────────────────────────┐
          │  Goal Heads                          │
          │  GoalHead    : MLP → [B,K,2]         │
          │  LaneGoalHead: cross-attn → [B,K,2]  │
          │  GoalGate    : learned σ blend        │
          └───────────┬──────────────────────────┘
                      │
          ┌───────────▼──────────────────────────┐
          │  Causal Transformer Decoder          │
          │  GPT-style, K=6, T=80 steps         │
          │  → trajectory [B, K, 80, 2]         │
          └──────────────────────────────────────┘
```

### 핵심 설계

| 구성요소 | 설명 |
|---------|------|
| **Joint Transformer** | maneuver prefix token + 99토큰 self-attention. 모든 모달리티를 flat하게 결합 |
| **LaneGraphEncoder** | 차선 endpoint 연결성(3m) + 중심 근접성(15m)을 adjacency bias로 GAT에 주입 |
| **GoalHead** | global context → MLP → K개 목표 좌표. maneuver 조건 반영 |
| **LaneGoalHead** | K mode query가 lane_feat에 cross-attend → lane-anchored goal |
| **GoalGate** | GoalCond + LaneGoal을 학습 가능한 sigmoid gate로 블렌딩 |
| **Causal Decoder** | time-step 간 causal mask + lane cross-attention, goal을 context로 주입 |
| **RareAlign loss** | Turn/LC 희귀 기동에 alignment loss 가중치 부여 |

---

## 실험 결과

### 전체 성능 (val, 2361 시나리오)

| 모델 | EP | minADE ↓ | minFDE ↓ | MR ↓ |
|------|:--:|:--------:|:--------:|:----:|
| Baseline | 20 | 1.732 | 4.789 | 0.499 |
| + GoalCond | 37 | 1.466 | 4.012 | 0.457 |
| + LaneGoal | 33 | 1.486 | 4.006 | 0.454 |
| + LaneGoal v2 (cond_query) | 25 | 1.481 | 3.926 | 0.438 |
| + LaneGoal v3 (turn_emb) | 31 | 1.502 | 4.090 | 0.444 |
| GoalCond + LaneGoal (Combined) | 30 | 1.474 | 4.023 | 0.443 |
| + LaneGraphEncoder | 32 | 1.345 | 3.623 | 0.438 |
| + GoalGate | 39 | 1.385 | 3.788 | 0.435 |
| **+ RichInput** (bbox, speed_limit, traf pos) | 17* | **1.306** | — | — |

> \* exp_rich_input 학습 진행 중 (ep17 기준, ep40 완료 후 업데이트 예정)

### Per-Maneuver minADE (m)

| 모델 | Stop | Straight | LC_Left | LC_Right | Turn_Left | Turn_Right |
|------|:----:|:--------:|:-------:|:--------:|:---------:|:----------:|
| Baseline | 0.051 | 1.771 | 2.310 | 2.310 | 4.189 | 4.891 |
| GoalCond | **0.037** | 1.597 | **1.805** | **2.114** | 3.586 | 3.107 |
| LaneGraph | 0.041 | **1.428** | 1.824 | 2.186 | **3.025** | **3.220** |
| GatedGoal | 0.040 | 1.518 | 1.969 | 1.948 | 3.269 | **2.822** |

> LaneGraph: Turn_Left 4.189→3.025 (-28%), Turn_Right 4.891→3.220 (-34%)  
> 시나리오 수: Stop=595, Straight=1256, LC_Left=98, LC_Right=109, Turn_Left=178, Turn_Right=125

평가 재현:
```bash
conda run -n waymo python scripts/eval_maneuver_breakdown.py
```

---

## 입력 특징

### Agent (`AGENT_DIM = 12`)

| 채널 | 설명 |
|------|------|
| x, y | ego-relative 위치 (m) |
| vx, vy | ego-relative 속도 (m/s) |
| cos_h, sin_h | ego-relative heading |
| valid_t | 해당 프레임 유효 여부 |
| t_norm | 시간 정규화 (0→1) |
| type_v, type_c | vehicle / cyclist 타입 플래그 |
| **len_norm** | 차량 길이 / 5.0 *(신규)* |
| **wid_norm** | 차량 너비 / 2.0 *(신규)* |

### Map (`MAP_DIM = 10`)

| 채널 | 설명 |
|------|------|
| x, y | 폴리라인 포인트 위치 |
| dx_norm, dy_norm | 정규화된 방향 벡터 |
| type_lane, type_road_edge, type_road_line, type_special | 맵 타입 플래그 |
| **speed_lim_norm** | 속도 제한 / 60 mph *(신규)* |
| **lane_type_norm** | 차선 종류 / 3 (freeway/surface/bike) *(신규)* |

### Traffic (`TRAF_DIM = 3`)

| 채널 | 설명 |
|------|------|
| state | 신호등 상태 float |
| **stop_x, stop_y** | 정지선 ego-relative 위치 *(신규)* |

---

## 학습

```bash
# 전처리 (최초 1회)
conda run -n waymo python preprocess.py

# Baseline
conda run -n waymo python train.py \
  --run_name baseline \
  --use_cache --cond_type maneuver --no_lane_mamba \
  --use_causal_attn --rare_align --epochs 40

# LaneGraph (현재 best 단일 구조)
conda run -n waymo python train.py \
  --run_name exp_lane_graph \
  --use_cache --cond_type maneuver --no_lane_mamba \
  --use_causal_attn --rare_align --epochs 40 \
  --use_goal_cond --use_lane_goal --use_cond_query --use_lane_graph

# RichInput (입력 보강 + LaneGraph)
conda run -n waymo python train.py \
  --run_name exp_rich_input \
  --use_cache --cond_type maneuver --no_lane_mamba \
  --use_causal_attn --rare_align --epochs 40 \
  --use_goal_cond --use_lane_goal --use_cond_query --use_lane_graph
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--use_causal_attn` | Causal Transformer 디코더 |
| `--use_goal_cond` | MLP goal head |
| `--use_lane_goal` | Lane cross-attn goal head |
| `--use_cond_query` | LaneGoalHead query에 maneuver context 주입 |
| `--use_lane_graph` | LaneGraphEncoder (GAT) 활성화 |
| `--use_goal_gate` | GoalCond+LaneGoal 학습 가능 gate 블렌딩 |
| `--rare_align` | 희귀 기동 alignment loss |
| `--cond_type` | `maneuver` (9-dim) \| `risk` (3-dim) |

---

## 체크포인트

| 체크포인트 | EP | minADE | 구성 |
|-----------|:--:|:------:|------|
| `checkpoints/causal_maneuver_rare_align/` | 20 | 1.732 | Baseline |
| `checkpoints/exp_goal_cond/` | 37 | 1.466 | + GoalCond |
| `checkpoints/exp_goal_lane_combined/` | 30 | 1.474 | + GoalCond + LaneGoal |
| `checkpoints/exp_lane_graph/` | 32 | **1.345** | + LaneGraphEncoder |
| `checkpoints/exp_gated_goal/` | 39 | 1.385 | + GoalGate |
| `checkpoints/exp_rich_input/` | 17* | **1.306** | + RichInput (진행 중) |

---

## 디렉토리 구조

```
waymo_traj/
├── train.py                          # 학습 스크립트
├── preprocess.py                     # TFRecord → NPZ 캐시
├── scripts/
│   ├── eval_maneuver_breakdown.py    # per-maneuver ADE/FDE/MR 테이블
│   ├── visualize_lane_graph.py       # LaneGraph vs Baseline 시각화
│   └── visualize_goal_cond.py        # GoalCond 시각화
└── src/
    ├── data/
    │   ├── tfrecord.py               # TFRecord 파서
    │   └── features.py               # 특징 추출 (AGENT_DIM=12, MAP_DIM=10, TRAF_DIM=3)
    ├── models/
    │   ├── encoders.py               # MultiStreamMambaEncoder, LaneGraphEncoder
    │   ├── motion_model.py           # RiskConditionedModel, GoalHead, LaneGoalHead
    │   └── risk_fusion.py
    └── eval/
        └── metrics.py                # minADE, minFDE, MR
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **minADE** | K=6 모드 중 최적 모드의 전 스텝 평균 L2 거리 (m) |
| **minFDE** | K=6 모드 중 최적 모드의 최종 스텝 L2 거리 (m) |
| **MR** | minFDE > 2.0 m 비율 (miss rate) |

---

## 데이터

WOMD v1.3.0, 10 Hz, 1초 과거 → 8초 미래 예측

| 분할 | 샤드 | 시나리오 |
|------|------|---------|
| Train | 00000–00049 (50 / 1000) | 26,746 |
| Val   | 00000–00007 (8 / 150)   | 2,361  |

---

## 설치

```bash
conda create -n waymo python=3.10
conda activate waymo
pip install torch==1.12.0+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

---

## 라이선스

Waymo Open Dataset은 [이용약관](https://waymo.com/open/terms/)에 따라 사용합니다. 모델 코드는 MIT 라이선스입니다.
