"""
LSTMBaseline 학습 스크립트.

train shards 00000-00049로 학습, val shards 00000-00007로 평가.
빠른 수렴을 위해 에폭 5, lr 1e-3.

Run:
    cd waymo_traj
    python train_lstm.py [--epochs 5] [--device cuda]
"""

import argparse
import os
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torch.nn.functional as F

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "waymo-motion-v1_3_0")
sys.path.insert(0, ROOT)

from src.data.tfrecord    import iter_tfrecords
from src.data.features    import extract_features
from src.models.baselines import LSTMBaseline
from src.eval.metrics     import compute_minADE_FDE

TRAIN_PATHS = [
    os.path.join(DATA_ROOT, "train", f"training.tfrecord-{i:05d}-of-01000")
    for i in range(50)
]
EVAL_PATHS = [
    os.path.join(DATA_ROOT, "val", f"validation.tfrecord-{i:05d}-of-00150")
    for i in range(8)
]
CKPT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=5)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--log_every",     type=int, default=200)
    p.add_argument("--max_scenarios", type=int, default=None,
                   help="에폭당 최대 시나리오 수 (빠른 테스트용)")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_one_epoch(model, optimizer, paths, device, train, log_every=200, max_scenarios=None):
    from waymo_open_dataset.protos import scenario_pb2

    total_loss = total_ade = total_fde = 0.0
    n_ok = 0
    t_start = time.time()

    for raw_bytes in iter_tfrecords(paths):
        if max_scenarios and n_ok >= max_scenarios:
            break

        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            feats = extract_features(sc)
        except Exception:
            continue
        if not feats["gt_valid"].any():
            continue

        # 입력: ego (x, y) 히스토리 [1, T, 2]
        ego_xy = torch.from_numpy(
            feats["agent_tensor"][0, :, :2]
        ).unsqueeze(0).to(device)                       # [1, 11, 2]

        gt_traj  = torch.from_numpy(feats["gt_trajectory"]).to(device)   # [80, 2]
        gt_valid = torch.from_numpy(feats["gt_valid"]).to(device)         # [80]
        valid_idx = gt_valid.nonzero(as_tuple=True)[0]

        pred = model(ego_xy)[0]                          # [80, 2]
        loss = F.l1_loss(pred[valid_idx], gt_traj[valid_idx])

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        with torch.no_grad():
            ade, fde = compute_minADE_FDE(
                pred.cpu().numpy(),
                feats["gt_trajectory"],
                feats["gt_valid"],
            )

        total_loss += loss.item()
        total_ade  += ade
        total_fde  += fde
        n_ok       += 1

        if n_ok % log_every == 0:
            mode = "train" if train else "eval"
            print(f"  [{mode}] {n_ok:5d}  "
                  f"loss={total_loss/n_ok:.4f}  "
                  f"ADE={total_ade/n_ok:.3f}m  "
                  f"FDE={total_fde/n_ok:.3f}m  "
                  f"({time.time()-t_start:.0f}s)")

    if n_ok == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return total_loss / n_ok, total_ade / n_ok, total_fde / n_ok, n_ok


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}  lr={args.lr}")
    print()

    model     = LSTMBaseline(hidden_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_ade = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"{'='*55}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*55}")

        model.train()
        tr_loss, tr_ade, tr_fde, n_tr = run_one_epoch(
            model, optimizer, TRAIN_PATHS, device, train=True,
            log_every=args.log_every, max_scenarios=args.max_scenarios,
        )
        print(f"  [train] loss={tr_loss:.4f}  ADE={tr_ade:.3f}m  FDE={tr_fde:.3f}m  ({n_tr})")

        model.eval()
        with torch.no_grad():
            ev_loss, ev_ade, ev_fde, n_ev = run_one_epoch(
                model, None, EVAL_PATHS, device, train=False,
                log_every=args.log_every, max_scenarios=args.max_scenarios,
            )
        print(f"  [eval]  loss={ev_loss:.4f}  ADE={ev_ade:.3f}m  FDE={ev_fde:.3f}m  ({n_ev})")

        scheduler.step()

        ckpt = os.path.join(CKPT_DIR, f"lstm_epoch{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "ev_ade": ev_ade, "ev_fde": ev_fde}, ckpt)
        print(f"  Saved → {ckpt}")

        if ev_ade < best_ade:
            best_ade = ev_ade
            best = os.path.join(CKPT_DIR, "lstm_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ev_ade": ev_ade, "ev_fde": ev_fde}, best)
            print(f"  ** New best ADE={ev_ade:.3f}m → {best}")
        print()

    print(f"LSTM 학습 완료.  Best ADE = {best_ade:.3f}m")


if __name__ == "__main__":
    main()
