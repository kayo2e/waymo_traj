"""
Training script for WaymoMotionModel.

Train : training.tfrecord-00000~00005-of-01000  (6 shards, ~3000 scenarios)
Eval  : training.tfrecord-00006~00007-of-01000  (2 shards, ~1000 scenarios)

Run:
    cd /home/dtlab/gy/waymo_traj
    python train.py [--epochs 10] [--lr 1e-4] [--device cuda]

TFRecord은 TF 없이 raw binary로 직접 파싱합니다 (protobuf 버전 충돌 우회).
"""

import argparse
import os
import sys
import time

# Must be set before any protobuf / waymo_open_dataset import
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torch.nn.functional as F

# ── paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "waymo-motion-v1_3_0/train")

sys.path.insert(0, ROOT)

from waymo_traj.src.data.tfrecord    import iter_tfrecords
from waymo_traj.src.data.features    import extract_features
from waymo_traj.src.models.motion_model import WaymoMotionModel


def _shard(n):
    return os.path.join(DATA_DIR, f"training.tfrecord-{n:05d}-of-01000")


TRAIN_PATHS = [_shard(i) for i in range(6)]    # 00000-00005
EVAL_PATHS  = [_shard(i) for i in range(6, 8)] # 00006-00007
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=10)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int,   default=50, help="print loss every N scenarios")
    p.add_argument("--resume",    type=str,   default=None, help="checkpoint path to resume from")
    return p.parse_args()


def to_tensor(arr, device):
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def run_one_epoch(model, optimizer, tfrecord_paths, device, train, log_every=50):
    """
    Stream scenarios from one or more tfrecord paths.
    Returns (avg_loss, avg_minADE, avg_minFDE, n_scenarios).
    """
    from waymo_open_dataset.protos import scenario_pb2

    total_loss = 0.0
    total_ade  = 0.0
    total_fde  = 0.0
    n_ok       = 0
    t_start    = time.time()

    for raw_bytes in iter_tfrecords(tfrecord_paths):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)

        try:
            feats = extract_features(sc)
        except Exception:
            continue

        if not feats["gt_valid"].any():
            continue

        agents  = to_tensor(feats["agent_tensor"],   device)   # [1,32,11,6]
        scene   = to_tensor(feats["scene_tensor"],   device)   # [1,50,10,3]
        traffic = to_tensor(feats["traffic_tensor"], device)   # [1,6,1]

        gt_kp    = torch.from_numpy(feats["gt_keypoints"]).to(device)   # [3,2]
        kp_valid = torch.from_numpy(feats["kp_valid"]).to(device)        # [3]
        gt_traj  = torch.from_numpy(feats["gt_trajectory"]).to(device)  # [80,2]
        gt_valid = torch.from_numpy(feats["gt_valid"]).to(device)        # [80]

        out       = model(agents, scene, traffic)
        pred_kp   = out["keypoints"][0]    # [K, 3, 2]
        pred_traj = out["trajectory"][0]   # [K, 80, 2]
        K         = pred_traj.shape[0]

        valid_idx = gt_valid.nonzero(as_tuple=True)[0]

        # Winner-Takes-All: pick mode closest to GT
        with torch.no_grad():
            mode_ades = torch.stack([
                torch.linalg.norm(pred_traj[k][valid_idx] - gt_traj[valid_idx], dim=1).mean()
                for k in range(K)
            ])
            best_k = int(mode_ades.argmin())

        if kp_valid.any():
            kp_loss = F.huber_loss(pred_kp[best_k][kp_valid], gt_kp[kp_valid], delta=2.0)
        else:
            kp_loss = pred_kp.sum() * 0.0

        traj_loss = F.l1_loss(pred_traj[best_k][valid_idx], gt_traj[valid_idx])
        loss      = kp_loss + traj_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        with torch.no_grad():
            all_dists = torch.stack([
                torch.linalg.norm(pred_traj[k][valid_idx] - gt_traj[valid_idx], dim=1)
                for k in range(K)
            ])                                         # [K, n_valid]
            ade = float(all_dists.mean(dim=1).min())   # minADE
            fde = float(all_dists[:, -1].min())        # minFDE

        total_loss += loss.item()
        total_ade  += ade
        total_fde  += fde
        n_ok       += 1

        if n_ok % log_every == 0:
            elapsed  = time.time() - t_start
            avg_loss = total_loss / n_ok
            avg_ade  = total_ade  / n_ok
            avg_fde  = total_fde  / n_ok
            mode_str = "train" if train else "eval"
            print(f"  [{mode_str}] {n_ok:4d} scenarios  "
                  f"loss={avg_loss:.4f}  minADE={avg_ade:.3f}m  minFDE={avg_fde:.3f}m  "
                  f"({elapsed:.0f}s)")

    if n_ok == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return total_loss / n_ok, total_ade / n_ok, total_fde / n_ok, n_ok


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Train  : shards 00000-00005  ({len(TRAIN_PATHS)} files)")
    print(f"Eval   : shards 00006-00007  ({len(EVAL_PATHS)} files)")
    print(f"Epochs : {args.epochs}  lr={args.lr}")
    print()

    model     = WaymoMotionModel(d_model=128, K=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}  (epoch {start_epoch})")

    best_eval_ade = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        model.train()
        tr_loss, tr_ade, tr_fde, n_tr = run_one_epoch(
            model, optimizer, TRAIN_PATHS, device, train=True, log_every=args.log_every
        )
        print(f"  [train] DONE  loss={tr_loss:.4f}  minADE={tr_ade:.3f}m  minFDE={tr_fde:.3f}m  ({n_tr} scenarios)")

        model.eval()
        with torch.no_grad():
            ev_loss, ev_ade, ev_fde, n_ev = run_one_epoch(
                model, None, EVAL_PATHS, device, train=False, log_every=args.log_every
            )
        print(f"  [eval]  DONE  loss={ev_loss:.4f}  minADE={ev_ade:.3f}m  minFDE={ev_fde:.3f}m  ({n_ev} scenarios)")

        scheduler.step()

        ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{epoch:02d}.pt")
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tr_loss":   tr_loss, "tr_ade": tr_ade, "tr_fde": tr_fde,
            "ev_loss":   ev_loss, "ev_ade": ev_ade, "ev_fde": ev_fde,
        }, ckpt_path)
        print(f"  Saved  → {ckpt_path}")

        if ev_ade < best_eval_ade:
            best_eval_ade = ev_ade
            best_path = os.path.join(CKPT_DIR, "model_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ev_ade": ev_ade, "ev_fde": ev_fde}, best_path)
            print(f"  ** New best  minADE={ev_ade:.3f}m  minFDE={ev_fde:.3f}m → {best_path}")

        print()

    print(f"Training complete.  Best eval minADE = {best_eval_ade:.3f}m")
    print(f"Best checkpoint    : {os.path.join(CKPT_DIR, 'model_best.pt')}")


if __name__ == "__main__":
    main()
