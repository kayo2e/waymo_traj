"""
위험 조건부 궤적 예측 학습 스크립트 (RiskConditionedModel).

Train : training.tfrecord-00000~00049-of-01000  (50 shards)
Eval  : validation.tfrecord-00000~00007-of-00150 (8 shards)

Run:
    cd waymo_traj
    python train.py [--epochs 50] [--lr 1e-4] [--device cuda]
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "waymo-motion-v1_3_0")
sys.path.insert(0, ROOT)

from src.data.tfrecord       import iter_tfrecords
from src.data.features       import extract_features, extract_risk_label
from src.models.motion_model import RiskConditionedModel
from src.eval.metrics        import compute_minADE_FDE, compute_MR


def _train_shard(n):
    return os.path.join(DATA_ROOT, "train", f"training.tfrecord-{n:05d}-of-01000")

def _val_shard(n):
    return os.path.join(DATA_ROOT, "val", f"validation.tfrecord-{n:05d}-of-00150")


TRAIN_PATHS = [_train_shard(i) for i in range(50)]   # 00000-00049
EVAL_PATHS  = [_val_shard(i)   for i in range(8)]    # 00000-00007
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--wd",        type=float, default=1e-4, help="weight decay")
    p.add_argument("--lam",       type=float, default=0.5,  help="L_risk 가중치")
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every",      type=int,  default=50)
    p.add_argument("--resume",         type=str,  default=None)
    p.add_argument("--max_scenarios",  type=int,  default=None,
                   help="에폭당 최대 시나리오 수 (빠른 테스트용)")
    return p.parse_args()


def _prep_inputs(feats: dict, device):
    """
    extract_features 출력을 RiskConditionedModel 입력 형식으로 변환.

    Returns (ego_hist, social_agents, map_tokens) — 각각 [1, *, *] 텐서
      ego_hist      [1, 11, 6]
      social_agents [1, 31, 6]  (T_HIST 평균)
      map_tokens    [1, 56, 3]  (map 50 + traffic 6, max-pool over pts)
    """
    agent = feats["agent_tensor"]    # [32, 11, 6]
    scene = feats["scene_tensor"]    # [50, 10, 3]
    traf  = feats["traffic_tensor"]  # [6, 1]

    ego_hist = agent[0:1]                      # [1, 11, 6]
    social   = agent[1:].mean(axis=1)[None]    # [1, 31, 6]

    map_poly  = scene.max(axis=1)              # [50, 3]
    traf_pad  = np.pad(traf, [(0, 0), (0, 2)])  # [6, 3]
    map_tok   = np.concatenate([map_poly, traf_pad], axis=0)[None]  # [1, 56, 3]

    return (
        torch.from_numpy(ego_hist).to(device),
        torch.from_numpy(social).to(device),
        torch.from_numpy(map_tok).to(device),
    )


def run_one_epoch(model, optimizer, tfrecord_paths, device, train,
                  lam=0.5, log_every=50, max_scenarios=None):
    """
    Returns (avg_loss, avg_minADE, avg_minFDE, avg_MR, n_scenarios)
    """
    from waymo_open_dataset.protos import scenario_pb2

    bce = nn.BCEWithLogitsLoss()

    total_loss = total_ade = total_fde = total_mr = 0.0
    n_ok = 0
    t_start = time.time()

    for raw_bytes in iter_tfrecords(tfrecord_paths):
        if max_scenarios and n_ok >= max_scenarios:
            break

        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)

        try:
            feats = extract_features(sc)
            risk_label_np = extract_risk_label(sc)           # [3]
        except Exception:
            continue

        if not feats["gt_valid"].any():
            continue

        ego_hist, social, map_tok = _prep_inputs(feats, device)

        risk_gt = torch.from_numpy(risk_label_np).unsqueeze(0).to(device)  # [1, 3]

        gt_kp    = torch.from_numpy(feats["gt_keypoints"]).to(device)  # [3, 2]
        kp_valid = torch.from_numpy(feats["kp_valid"]).to(device)      # [3]
        gt_traj  = torch.from_numpy(feats["gt_trajectory"]).to(device) # [80, 2]
        gt_valid = torch.from_numpy(feats["gt_valid"]).to(device)      # [80]

        # ── forward ──────────────────────────────────────────────────────────
        out       = model(ego_hist, social, map_tok, risk_label=risk_gt if train else None)
        pred_kp   = out["keypoints"][0]     # [K, 3, 2]
        pred_traj = out["trajectory"][0]    # [K, 80, 2]
        risk_logits = out["risk_logits"][0] # [3]
        K = pred_traj.shape[0]

        valid_idx = gt_valid.nonzero(as_tuple=True)[0]

        # ── Winner-Takes-All ─────────────────────────────────────────────────
        with torch.no_grad():
            mode_ades = torch.stack([
                torch.linalg.norm(pred_traj[k][valid_idx] - gt_traj[valid_idx], dim=1).mean()
                for k in range(K)
            ])
            best_k = int(mode_ades.argmin())

        # ── L_traj ───────────────────────────────────────────────────────────
        if kp_valid.any():
            kp_loss = F.huber_loss(pred_kp[best_k][kp_valid], gt_kp[kp_valid], delta=2.0)
        else:
            kp_loss = pred_kp.sum() * 0.0

        traj_loss = F.l1_loss(pred_traj[best_k][valid_idx], gt_traj[valid_idx])
        L_traj = kp_loss + traj_loss

        # ── L_risk (BCE) ──────────────────────────────────────────────────────
        L_risk = bce(risk_logits, risk_gt[0])

        loss = L_traj + lam * L_risk

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # ── 지표 ──────────────────────────────────────────────────────────────
        with torch.no_grad():
            traj_np  = pred_traj.cpu().numpy()
            gt_np    = feats["gt_trajectory"]
            valid_np = feats["gt_valid"]
            ade, fde = compute_minADE_FDE(traj_np, gt_np, valid_np)
            mr       = compute_MR(traj_np, gt_np, valid_np)

        total_loss += loss.item()
        total_ade  += ade
        total_fde  += fde
        total_mr   += mr
        n_ok       += 1

        if n_ok % log_every == 0:
            elapsed  = time.time() - t_start
            mode_str = "train" if train else "eval"
            print(f"  [{mode_str}] {n_ok:4d} scenarios  "
                  f"loss={total_loss/n_ok:.4f}  "
                  f"minADE={total_ade/n_ok:.3f}m  "
                  f"minFDE={total_fde/n_ok:.3f}m  "
                  f"MR={total_mr/n_ok:.3f}  "
                  f"({elapsed:.0f}s)")

    if n_ok == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    return (total_loss / n_ok, total_ade / n_ok,
            total_fde / n_ok, total_mr / n_ok, n_ok)


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Train  : training   shards 00000-00049  ({len(TRAIN_PATHS)} files)")
    print(f"Eval   : validation shards 00000-00007  ({len(EVAL_PATHS)} files)")
    print(f"Epochs : {args.epochs}  lr={args.lr}  wd={args.wd}  λ={args.lam}")
    print()

    model     = RiskConditionedModel(d_model=128, K=6, n_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}  (epoch {start_epoch})")

    best_eval_ade = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"{'='*65}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*65}")

        model.train()
        tr_loss, tr_ade, tr_fde, tr_mr, n_tr = run_one_epoch(
            model, optimizer, TRAIN_PATHS, device, train=True,
            lam=args.lam, log_every=args.log_every,
            max_scenarios=args.max_scenarios,
        )
        print(f"  [train] DONE  loss={tr_loss:.4f}  "
              f"minADE={tr_ade:.3f}m  minFDE={tr_fde:.3f}m  MR={tr_mr:.3f}  "
              f"({n_tr} scenarios)")

        model.eval()
        with torch.no_grad():
            ev_loss, ev_ade, ev_fde, ev_mr, n_ev = run_one_epoch(
                model, None, EVAL_PATHS, device, train=False,
                lam=args.lam, log_every=args.log_every,
                max_scenarios=args.max_scenarios,
            )
        print(f"  [eval]  DONE  loss={ev_loss:.4f}  "
              f"minADE={ev_ade:.3f}m  minFDE={ev_fde:.3f}m  MR={ev_mr:.3f}  "
              f"({n_ev} scenarios)")

        scheduler.step()

        ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{epoch:02d}.pt")
        torch.save({
            "epoch":    epoch,
            "model":    model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tr_loss": tr_loss, "tr_ade": tr_ade, "tr_fde": tr_fde, "tr_mr": tr_mr,
            "ev_loss": ev_loss, "ev_ade": ev_ade, "ev_fde": ev_fde, "ev_mr": ev_mr,
        }, ckpt_path)
        print(f"  Saved  → {ckpt_path}")

        if ev_ade < best_eval_ade:
            best_eval_ade = ev_ade
            best_path = os.path.join(CKPT_DIR, "model_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ev_ade": ev_ade, "ev_fde": ev_fde, "ev_mr": ev_mr},
                       best_path)
            print(f"  ** New best  minADE={ev_ade:.3f}m  "
                  f"minFDE={ev_fde:.3f}m  MR={ev_mr:.3f}  → {best_path}")

        print()

    print(f"학습 완료.  Best eval minADE = {best_eval_ade:.3f}m")
    print(f"Best checkpoint : {os.path.join(CKPT_DIR, 'model_best.pt')}")


if __name__ == "__main__":
    main()
