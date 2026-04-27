# Waymo Trajectory Prediction with Mamba

Autonomous vehicle trajectory prediction on the [Waymo Open Motion Dataset (WOMD) v1.3.0](https://waymo.com/open/data/motion/), using a 3-stage pipeline: risk-aware scene understanding → Mamba-based multi-agent encoding → Gemini-assisted trajectory refinement.

## Overview

```
Stage 1  Scene parsing & risk event labeling
  ↓
Stage 2  Multi-agent Mamba encoding → context tokens
  ↓
Stage 3  K=6 diverse trajectory hypotheses + Gemini explanation
```

**Key design choices**

- **No TensorFlow at inference time** — TFRecords are parsed with raw binary reads (`struct.unpack`), sidestepping the TF/protobuf version conflict entirely.
- **Mamba SSM** instead of Transformer attention — linear-time sequence modeling over 88 scene tokens (32 agents + 50 map polylines + 6 traffic signals).
- **K=6 multi-modal prediction** with Winner-Takes-All (WTA) training loss.
- **Gemini integration** for natural-language trajectory explanation and physical-consistency refinement.

---

## Architecture

### Token layout (88 tokens × 128-D)

| Index | Content |
|-------|---------|
| 0 | Ego agent |
| 1–31 | Nearby agents (sorted by distance) |
| 32–81 | Map polylines |
| 82–87 | Traffic signals |

### Model components

```
agents  [B, 32, 11, 6]  ──┐
map     [B, 50, 10, 3]  ──┤ JointPolylineEncoder  →  [B, N, 128]
traffic [B,  6,  1]     ──┘

concat → [B, 88, 128]
         ↓
    TambaMambaEncoder (2-layer Mamba, d_model=128)
         ↓
  context [B, 88, 128]
         ↓  ego token [B, 128]
         ×  K mode queries [K, 128]          ← nn.Embedding(6, 128)
         ↓
  KeypointDecoder   → keypoints  [B, K, 3, 2]   (1s / 3s / 5s)
  TrajectoryRefiner → trajectory [B, K, 80, 2]  (8s @ 10Hz)
```

**JointPolylineEncoder**: `Linear → LayerNorm → ReLU → Linear` + max-pool over point dimension for polyline inputs.

**TambaMambaEncoder**: wraps HuggingFace `MambaModel` with `n_layers=2, expand=2, d_conv=4, d_state=16`.

**Winner-Takes-All loss**: only the mode with the lowest ADE to ground truth receives gradient — promotes diversity across the K hypotheses.

---

## Repository Structure

```
waymo_traj/
├── train.py                  # Training entry point
├── evaluate.py               # Baseline comparison table
├── requirements.txt
├── notebooks/
│   └── pipeline.ipynb        # End-to-end demo notebook
└── src/
    ├── data/
    │   ├── tfrecord.py       # TFRecord iterator (no TF)
    │   └── features.py       # Scenario → feature tensors
    ├── models/
    │   ├── encoders.py       # TambaMambaEncoder, JointPolylineEncoder
    │   ├── motion_model.py   # WaymoMotionModel (K=6)
    │   └── baselines.py      # ConstantVelocityBaseline, LSTMBaseline
    ├── pipeline/
    │   ├── stage1.py         # Risk event detection
    │   ├── stage2.py         # Mamba encoding bridge
    │   └── stage3_gemini.py  # Gemini explanation & refinement
    ├── eval/
    │   └── metrics.py        # minADE, minFDE
    └── viz/
        └── plot.py           # Trajectory visualization
```

---

## Data

WOMD v1.3.0 — download from `gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/`

| Split | Shards used | Location |
|-------|-------------|----------|
| Train | 00000–00049 (50 / 1000) | `waymo-motion-v1_3_0/train/` |
| Val   | 00000–00007 (8 / 150)   | `waymo-motion-v1_3_0/val/`   |
| Test  | 00000–00007 (8 / 150)   | `waymo-motion-v1_3_0/test/`  |

<details>
<summary>Download commands (gsutil)</summary>

```bash
DATA=waymo-motion-v1_3_0
BASE=gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario

# Train (50 shards)
gsutil -m cp \
  $(for i in $(seq 0 49); do
      echo "$BASE/training/training.tfrecord-$(printf '%05d' $i)-of-01000"
    done) $DATA/train/

# Val (8 shards)
gsutil -m cp \
  $(for i in $(seq 0 7); do
      echo "$BASE/validation/validation.tfrecord-$(printf '%05d' $i)-of-00150"
    done) $DATA/val/

# Test (8 shards)
gsutil -m cp \
  $(for i in $(seq 0 7); do
      echo "$BASE/testing/testing.tfrecord-$(printf '%05d' $i)-of-00150"
    done) $DATA/test/
```

</details>

---

## Installation

```bash
conda create -n waymo python=3.10
conda activate waymo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install waymo-open-dataset-tf-2-11-0
```

> **Protobuf note**: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set automatically by `train.py` and `evaluate.py`. For the notebook, the kernel config in `.local/share/jupyter/kernels/waymo_kernel/kernel.json` sets it via the `env` field.

---

## Usage

### Training

```bash
cd waymo_traj
python train.py --epochs 10 --lr 1e-4 --device cuda
```

Checkpoints are saved to `checkpoints/model_epoch{N:02d}.pt`. The best validation minADE checkpoint is written to `checkpoints/model_best.pt`.

Optional flags:

```
--epochs     int   default 10
--lr         float default 1e-4
--device     str   default cuda (falls back to cpu)
--log_every  int   print interval in scenarios (default 50)
--resume     str   path to checkpoint to resume from
```

### Evaluation

```bash
python evaluate.py --ckpt checkpoints/model_best.pt --device cuda
```

Prints a minADE / minFDE comparison table across three models:

```
==============================================================
  Baseline Comparison vs WOMD Ground Truth
==============================================================
  Model                          minADE (m)   minFDE (m)
  ------------------------------  ----------   ----------
  Constant Velocity                    x.xxx        x.xxx
  LSTM (untrained)                     x.xxx        x.xxx
  WaymoMotionModel K=6                 x.xxx        x.xxx
==============================================================
```

### Notebook

Open `notebooks/pipeline.ipynb` with the `waymo_kernel` Jupyter kernel for the full interactive pipeline, including Gemini-generated explanations and trajectory visualization.

---

## Metrics

- **minADE** — minimum (over K modes) of mean L2 distance to ground truth across all valid future timesteps
- **minFDE** — minimum (over K modes) of L2 distance at the final valid timestep

Both are computed by `src/eval/metrics.py:compute_minADE_FDE`.

---

## Stage 3: Gemini Integration

`src/pipeline/stage3_gemini.py` provides:

| Function | Description |
|----------|-------------|
| `generate_gemini_explanation` | Returns a `[Subject]+[Action]+[Constraint]` safety narrative given risk labels and predicted keypoints |
| `mamba_context_to_text` | Summarizes Mamba attention norms as natural language |
| `gemini_refine_trajectory` | Asks Gemini to output a physically consistent 80-step trajectory override |

Models tried in order: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.0-flash-lite`.

Set `GOOGLE_API_KEY` environment variable before use.

---

## License

This project uses the Waymo Open Dataset under its [Terms of Service](https://waymo.com/open/terms/). Model code is MIT licensed.
