"""
Evaluation script: WaymoMotionModel vs Constant Velocity vs LSTM baselines.

Run:
    cd /home/dtlab/gy/waymo_traj
    python evaluate.py [--ckpt checkpoints/model_best.pt] [--device cuda]
"""

import argparse
import os
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from waymo_traj.src.data.tfrecord       import iter_tfrecords
from waymo_traj.src.data.features       import extract_features
from waymo_traj.src.models.motion_model import WaymoMotionModel
from waymo_traj.src.models.baselines    import ConstantVelocityBaseline, LSTMBaseline
from waymo_traj.src.eval.metrics        import compute_minADE_FDE


def _shard(n):
    return os.path.join(ROOT, "waymo-motion-v1_3_0/train",
                        f"training.tfrecord-{n:05d}-of-01000")


EVAL_PATHS = [_shard(i) for i in range(6, 8)]   # 00006-00007


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   type=str, default=os.path.join(ROOT, "checkpoints/model_best.pt"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_scenarios", type=int, default=None,
                   help="limit number of scenarios (None = all)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Eval   : shards 00006-00007  ({len(EVAL_PATHS)} files)")
    print(f"Ckpt   : {args.ckpt}")
    print()

    # ── Load WaymoMotionModel ────────────────────────────────────────────────
    model = WaymoMotionModel(d_model=128, K=6).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[Checkpoint loaded] epoch={ckpt.get('epoch','?')}  "
              f"ev_minADE={ckpt.get('ev_ade', float('nan')):.3f}m")
    else:
        print(f"[WARNING] Checkpoint not found — using random init")
    model.eval()

    # ── Baselines ────────────────────────────────────────────────────────────
    cv_model   = ConstantVelocityBaseline()
    torch.manual_seed(0)
    lstm_model = LSTMBaseline().eval()

    # ── Accumulate metrics ───────────────────────────────────────────────────
    results = {
        "Constant Velocity":        {"ade": 0.0, "fde": 0.0, "n": 0},
        "LSTM (untrained)":         {"ade": 0.0, "fde": 0.0, "n": 0},
        "WaymoMotionModel K=6":     {"ade": 0.0, "fde": 0.0, "n": 0},
    }

    from waymo_open_dataset.protos import scenario_pb2
    t_start = time.time()

    for i, raw_bytes in enumerate(iter_tfrecords(EVAL_PATHS)):
        if args.max_scenarios and i >= args.max_scenarios:
            break

        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            feats = extract_features(sc)
        except Exception:
            continue
        if not feats["gt_valid"].any():
            continue

        gt_traj  = feats["gt_trajectory"]
        gt_valid = feats["gt_valid"]

        # Constant Velocity
        cv_pred = cv_model.predict(feats["agent_tensor"][0])
        cv_ade, cv_fde = compute_minADE_FDE(cv_pred, gt_traj, gt_valid)
        results["Constant Velocity"]["ade"] += cv_ade
        results["Constant Velocity"]["fde"] += cv_fde
        results["Constant Velocity"]["n"]   += 1

        # LSTM (untrained)
        ego_xy = torch.from_numpy(feats["agent_tensor"][0, :, :2]).unsqueeze(0)
        with torch.no_grad():
            lstm_pred = lstm_model(ego_xy)[0].numpy()
        l_ade, l_fde = compute_minADE_FDE(lstm_pred, gt_traj, gt_valid)
        results["LSTM (untrained)"]["ade"] += l_ade
        results["LSTM (untrained)"]["fde"] += l_fde
        results["LSTM (untrained)"]["n"]   += 1

        # WaymoMotionModel
        agents  = torch.from_numpy(feats["agent_tensor"]).unsqueeze(0).to(device)
        scene   = torch.from_numpy(feats["scene_tensor"]).unsqueeze(0).to(device)
        traffic = torch.from_numpy(feats["traffic_tensor"]).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(agents, scene, traffic)
        ml_traj = out["trajectory"][0].cpu().numpy()   # [K, 80, 2]
        m_ade, m_fde = compute_minADE_FDE(ml_traj, gt_traj, gt_valid)
        results["WaymoMotionModel K=6"]["ade"] += m_ade
        results["WaymoMotionModel K=6"]["fde"] += m_fde
        results["WaymoMotionModel K=6"]["n"]   += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1} scenarios  ({time.time()-t_start:.0f}s)")

    # ── Print table ──────────────────────────────────────────────────────────
    n_total = results["WaymoMotionModel K=6"]["n"]
    print(f"\nEvaluated {n_total} scenarios  ({time.time()-t_start:.0f}s total)\n")
    print("=" * 62)
    print("  Baseline Comparison vs WOMD Ground Truth")
    print("=" * 62)
    print(f"  {'모델':30} {'minADE (m)':>12} {'minFDE (m)':>12}")
    print(f"  {'-'*30} {'----------':>12} {'----------':>12}")
    for name, acc in results.items():
        n = acc["n"]
        if n == 0:
            continue
        ade = acc["ade"] / n
        fde = acc["fde"] / n
        print(f"  {name:30} {ade:>12.3f} {fde:>12.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
