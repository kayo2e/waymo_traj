"""
베이스라인 비교 평가 스크립트.

모델 3종 비교:
  1. Constant Velocity      (비학습)
  2. LSTM Baseline          (학습됨, --lstm_ckpt 지정)
  3. RiskConditionedModel   (학습됨, --ckpt 지정)

Run:
    cd waymo_traj
    python evaluate.py \
        --ckpt      checkpoints/model_best.pt \
        --lstm_ckpt checkpoints/lstm_best.pt \
        [--device cuda] [--max_scenarios 500]
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.tfrecord       import iter_tfrecords
from src.data.features       import extract_features
from src.models.motion_model import RiskConditionedModel
from src.models.baselines    import ConstantVelocityBaseline, LSTMBaseline
from src.eval.metrics        import compute_minADE_FDE, compute_MR


def _prep_inputs(feats, device):
    """agent_tensor / scene_tensor / traffic_tensor → RiskConditionedModel 입력 형식."""
    agent = feats["agent_tensor"]    # [32, 11, 6]
    scene = feats["scene_tensor"]    # [50, 10, 3]
    traf  = feats["traffic_tensor"]  # [6, 1]

    ego_hist = agent[0:1]                          # [1, 11, 6]
    social   = agent[1:].mean(axis=1)[None]        # [1, 31, 6]
    map_poly = scene.max(axis=1)                   # [50, 3]
    traf_pad = np.pad(traf, [(0, 0), (0, 2)])      # [6, 3]
    map_tok  = np.concatenate([map_poly, traf_pad], axis=0)[None]  # [1, 56, 3]

    return (
        torch.from_numpy(ego_hist).to(device),
        torch.from_numpy(social).to(device),
        torch.from_numpy(map_tok).to(device),
    )


def _val_shard(n):
    return os.path.join(ROOT, "waymo-motion-v1_3_0", "val",
                        f"validation.tfrecord-{n:05d}-of-00150")


EVAL_PATHS = [_val_shard(i) for i in range(8)]   # 00000-00007


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          type=str,
                   default=os.path.join(ROOT, "checkpoints/model_best.pt"))
    p.add_argument("--lstm_ckpt",     type=str,
                   default=os.path.join(ROOT, "checkpoints/lstm_best.pt"))
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_scenarios", type=int, default=None)
    return p.parse_args()


def _empty():
    return {"ade": 0.0, "fde": 0.0, "mr": 0.0, "n": 0}


def _update(acc, ade, fde, mr):
    acc["ade"] += ade
    acc["fde"] += fde
    acc["mr"]  += mr
    acc["n"]   += 1


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Eval   : val shards 00000-00007  ({len(EVAL_PATHS)} files)")
    print()

    # ── RiskConditionedModel ──────────────────────────────────────────────────
    mm_model = RiskConditionedModel(d_model=128, K=6, n_layers=2).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        mm_model.load_state_dict(ckpt["model"])
        ev_ade = ckpt.get("ev_ade", float("nan"))
        print(f"[RiskConditionedModel]  체크포인트 로드  "
              f"epoch={ckpt.get('epoch','?')}  val_minADE={ev_ade:.3f}m")
    else:
        print(f"[RiskConditionedModel]  체크포인트 없음 — random init  ({args.ckpt})")
    mm_model.eval()

    # ── LSTM Baseline ─────────────────────────────────────────────────────────
    lstm_model = LSTMBaseline(hidden_size=64, num_layers=2).to(device)
    lstm_trained = False
    if os.path.exists(args.lstm_ckpt):
        ckpt_l = torch.load(args.lstm_ckpt, map_location=device)
        lstm_model.load_state_dict(ckpt_l["model"])
        lstm_trained = True
        print(f"[LSTM Baseline]     체크포인트 로드  "
              f"epoch={ckpt_l.get('epoch','?')}  val_ADE={ckpt_l.get('ev_ade',float('nan')):.3f}m")
    else:
        print(f"[LSTM Baseline]     체크포인트 없음 — random init  ({args.lstm_ckpt})")
        print(f"                    → train_lstm.py 먼저 실행 권장")
    lstm_model.eval()

    cv_model = ConstantVelocityBaseline()

    lstm_label = "LSTM (trained)" if lstm_trained else "LSTM (untrained)"

    results = {
        "Constant Velocity":       _empty(),
        lstm_label:                _empty(),
        "RiskConditionedModel K=6": _empty(),
    }

    from waymo_open_dataset.protos import scenario_pb2
    t_start = time.time()
    print()

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

        # ── Constant Velocity ─────────────────────────────────────────────────
        cv_pred = cv_model.predict(feats["agent_tensor"][0])
        ade, fde = compute_minADE_FDE(cv_pred, gt_traj, gt_valid)
        mr       = compute_MR(cv_pred, gt_traj, gt_valid)
        _update(results["Constant Velocity"], ade, fde, mr)

        # ── LSTM ──────────────────────────────────────────────────────────────
        ego_xy = torch.from_numpy(
            feats["agent_tensor"][0, :, :2]
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            lstm_pred = lstm_model(ego_xy)[0].cpu().numpy()   # [80, 2]
        ade, fde = compute_minADE_FDE(lstm_pred, gt_traj, gt_valid)
        mr       = compute_MR(lstm_pred, gt_traj, gt_valid)
        _update(results[lstm_label], ade, fde, mr)

        # ── RiskConditionedModel ──────────────────────────────────────────────
        ego_hist, social, map_tok = _prep_inputs(feats, device)
        with torch.no_grad():
            out = mm_model(ego_hist, social, map_tok, risk_label=None)
        mm_pred = out["trajectory"][0].cpu().numpy()           # [K, 80, 2]
        ade, fde = compute_minADE_FDE(mm_pred, gt_traj, gt_valid)
        mr       = compute_MR(mm_pred, gt_traj, gt_valid)
        _update(results["RiskConditionedModel K=6"], ade, fde, mr)

        if (i + 1) % 200 == 0:
            print(f"  {i+1} scenarios  ({time.time()-t_start:.0f}s)")

    # ── 결과 테이블 ───────────────────────────────────────────────────────────
    n_total = results["RiskConditionedModel K=6"]["n"]
    print(f"\n평가 시나리오: {n_total}개  ({time.time()-t_start:.0f}s)\n")

    w = 26
    print("=" * 66)
    print("  Baseline Comparison vs WOMD Ground Truth  (test set)")
    print("=" * 66)
    print(f"  {'모델':{w}}  {'minADE (m)':>10}  {'minFDE (m)':>10}  {'MR (2m)':>8}")
    print(f"  {'-'*w}  {'----------':>10}  {'----------':>10}  {'-------':>8}")
    for name, acc in results.items():
        n = acc["n"]
        if n == 0:
            continue
        print(f"  {name:{w}}  {acc['ade']/n:>10.3f}  {acc['fde']/n:>10.3f}  {acc['mr']/n:>8.3f}")
    print("=" * 66)
    print()
    print("* MR: Miss Rate — minFDE > 2.0m 비율 (낮을수록 좋음)")


if __name__ == "__main__":
    main()
