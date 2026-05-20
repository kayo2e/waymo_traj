"""
위험 조건부 궤적 예측 학습 스크립트 (RiskConditionedModel).

Train : training.tfrecord-00000~00049-of-01000  (50 shards)
Eval  : validation.tfrecord-00000~00007-of-00150 (8 shards)

Run:
    cd waymo_traj
    python train.py [--epochs 50] [--lr 1e-4] [--device cuda]
"""

import argparse
import glob
import math
import os
import random
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
    p.add_argument("--run_name",  type=str,   default="baseline",
                   help="실험 이름 → checkpoints/<run_name>/ 에 저장")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--wd",        type=float, default=1e-4, help="weight decay")
    p.add_argument("--lam",       type=float, default=0.5,  help="L_risk 가중치")
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every",      type=int,  default=50)
    p.add_argument("--resume",         type=str,  default=None)
    p.add_argument("--max_scenarios",  type=int,  default=None,
                   help="에폭당 최대 시나리오 수 (빠른 테스트용)")
    p.add_argument("--loss",      type=str,   default="laplace",
                   choices=["wta", "laplace"],
                   help="wta: soft-WTA (non-best=0.3) | laplace: Laplace GMM NLL")
    p.add_argument("--laplace_b", type=float, default=1.0,
                   help="Laplace 스케일 b (--loss laplace 시 사용)")
    p.add_argument("--no_lane_mamba", action="store_true",
                   help="RiskAwareLaneMamba 비활성화 (ablation)")
    p.add_argument("--no_risk_prefix", action="store_true",
                   help="Risk prefix token 비활성화 → risk_head+RiskFusion 방식 (ablation)")
    p.add_argument("--use_traj_gpt", action="store_true",
                   help="TrajGPT (GPT-style causal 궤적 예측) 사용")
    p.add_argument("--gru", action="store_true",
                   help="TrajGPT 디코더를 GRU로 교체 (--use_traj_gpt 와 함께 사용)")
    p.add_argument("--no_traj_fix", action="store_true",
                   help="time_emb/vel_proj 비활성화 → baseline 아키텍처")
    p.add_argument("--use_cache", action="store_true",
                   help="preprocess.py로 생성한 .npz 캐시 사용 (빠른 학습)")
    p.add_argument("--cache_dir", type=str, default=os.path.join(ROOT, "cache"),
                   help="캐시 디렉토리 경로")
    return p.parse_args()


def _prep_inputs(feats: dict, device):
    """
    extract_features 출력을 RiskConditionedModel 입력 형식으로 변환.

    Returns (ego_hist, social_agents, map_scene, traf)
      ego_hist      [1, 11, 10]      에고 시계열
      social_agents [1, 31, 11, 10]  에이전트별 전체 시계열 (JointPolylineEncoder용)
      map_scene     [1, 50, 10, 6]   차선별 폴리라인 포인트 (JointPolylineEncoder용)
      traf          [1, 6,  1]       신호등 상태
    """
    agent = feats["agent_tensor"]    # [32, 11, 10]
    scene = feats["scene_tensor"]    # [50, 10, 6]
    traf  = feats["traffic_tensor"]  # [6, 1]

    ego_hist  = agent[0:1]           # [1, 11, 10]
    social    = agent[1:][None]      # [1, 31, 11, 10]  — 시계열 보존
    map_scene = scene[None]          # [1, 50, 10, 6]   — 폴리라인 형상 보존
    traf_t    = traf[None]           # [1, 6, 1]

    return (
        torch.from_numpy(ego_hist).to(device),
        torch.from_numpy(social).to(device),
        torch.from_numpy(map_scene).to(device),
        torch.from_numpy(traf_t).to(device),
    )


def run_one_epoch_cache(model, optimizer, npz_paths, device, train,
                        log_every=50, max_scenarios=None, tf_prob=1.0,
                        loss_mode="laplace", laplace_b=1.0, use_risk_prefix=True):
    """캐시(.npz) 기반 epoch — TFRecord 파싱 없이 빠른 학습."""
    bce = nn.BCEWithLogitsLoss()
    total_loss = total_ade = total_fde = total_mr = 0.0
    n_ok = 0

    paths = sorted(npz_paths)
    if train:
        random.shuffle(paths)

    mode_str = "train" if train else "eval"
    pbar = tqdm(paths, desc=f"{mode_str} shards", unit="shard")

    for npz_path in pbar:
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        N = len(data["agents"])
        idx_list = list(range(N))
        if train:
            random.shuffle(idx_list)

        for idx in idx_list:
            if max_scenarios and n_ok >= max_scenarios:
                break

            agent       = data["agents"][idx]       # [32,11,10]
            scene       = data["scenes"][idx]       # [50,10,6]
            traf        = data["trafs"][idx]        # [6,1]
            gt_traj_np  = data["gt_trajs"][idx]     # [80,2]
            gt_valid_np = data["gt_valids"][idx]    # [80]
            risk_np     = data["risk_labels"][idx]  # [3]

            if not gt_valid_np.any():
                continue

            ego_hist  = torch.from_numpy(agent[0:1]).to(device)
            social    = torch.from_numpy(agent[1:][None]).to(device)
            map_scene = torch.from_numpy(scene[None]).to(device)
            traf_t    = torch.from_numpy(traf[None]).to(device)
            risk_gt   = torch.from_numpy(risk_np).unsqueeze(0).to(device)
            gt_traj   = torch.from_numpy(gt_traj_np).to(device)
            gt_valid  = torch.from_numpy(gt_valid_np).to(device)

            out = model(ego_hist, social, map_scene, traf_t,
                        risk_label=risk_gt,
                        gt_traj=gt_traj.unsqueeze(0) if train else None,
                        tf_prob=tf_prob if train else 0.0)

            pred_traj   = out["trajectory"][0]
            risk_logits = out["risk_logits"]
            K = pred_traj.shape[0]
            valid_idx = gt_valid.nonzero(as_tuple=True)[0]

            traj_losses = torch.stack([
                F.l1_loss(pred_traj[k][valid_idx], gt_traj[valid_idx])
                for k in range(K)
            ])

            if loss_mode == "laplace":
                log_liks = -traj_losses / laplace_b
                L_traj = math.log(K) - torch.logsumexp(log_liks, dim=0)
            else:
                with torch.no_grad():
                    best_k = int(traj_losses.argmin())
                wta_weights = torch.full((K,), 0.3, device=device)
                wta_weights[best_k] = 1.0
                L_traj = (traj_losses * wta_weights).sum()

            if not use_risk_prefix and risk_logits is not None:
                loss = L_traj + 0.5 * bce(risk_logits[0], risk_gt[0])
            else:
                loss = L_traj

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            with torch.no_grad():
                traj_np = pred_traj.cpu().numpy()
                ade, fde = compute_minADE_FDE(traj_np, gt_traj_np, gt_valid_np)
                mr       = compute_MR(traj_np, gt_traj_np, gt_valid_np)

            total_loss += loss.item()
            total_ade  += ade
            total_fde  += fde
            total_mr   += mr
            n_ok       += 1

        pbar.set_postfix(
            n=n_ok,
            loss=f"{total_loss/max(n_ok,1):.4f}",
            ADE=f"{total_ade/max(n_ok,1):.3f}",
        )
        if max_scenarios and n_ok >= max_scenarios:
            break

    if n_ok == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    return total_loss/n_ok, total_ade/n_ok, total_fde/n_ok, total_mr/n_ok, n_ok


def run_one_epoch(model, optimizer, tfrecord_paths, device, train,
                  lam=0.5, log_every=50, max_scenarios=None, tf_prob=1.0,
                  loss_mode="laplace", laplace_b=1.0, use_risk_prefix=True):
    """
    Returns (avg_loss, avg_minADE, avg_minFDE, avg_MR, n_scenarios)
    """
    from waymo_open_dataset.protos import scenario_pb2

    bce = nn.BCEWithLogitsLoss()

    total_loss = total_ade = total_fde = total_mr = 0.0
    total_lp_max_risk = total_lp_max_norisk = 0.0
    n_risk = n_norisk = 0
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

        ego_hist, social, map_scene, traf = _prep_inputs(feats, device)

        risk_gt  = torch.from_numpy(risk_label_np).unsqueeze(0).to(device)  # [1, 3]
        gt_traj  = torch.from_numpy(feats["gt_trajectory"]).to(device)    # [80, 2]
        gt_valid = torch.from_numpy(feats["gt_valid"]).to(device)          # [80]

        # ── forward ───────────────────────────────────────────────────────────
        from src.models.traj_gpt import TrajGPT
        if isinstance(model, TrajGPT):
            # TrajGPT: gt_future는 (x,y)만 사용, NaN → 0 마스킹
            if train:
                gt_xy = gt_traj.unsqueeze(0).clone()               # [1, 80, 2]
                gt_xy[torch.isnan(gt_xy)] = 0.0
            else:
                gt_xy = None
            out = model(
                ego_hist, social, map_scene, traf,
                gt_future  = gt_xy,
                tf_prob    = tf_prob if train else 0.0,
                risk_label = risk_gt,
            )
        else:
            out = model(
                ego_hist, social, map_scene, traf,
                risk_label = risk_gt,
                gt_traj    = gt_traj.unsqueeze(0) if train else None,
                tf_prob    = tf_prob if train else 0.0,
            )
        pred_traj   = out["trajectory"][0]    # [K, 80, 2]
        risk_logits = out["risk_logits"]      # [1, 3] or None
        K = pred_traj.shape[0]

        # ── lane_prob 통계 누적 (use_lane_mamba=True 시) ──────────────────────
        if out["lane_prob"] is not None:
            lp_max = out["lane_prob"][0].max().item()
            if risk_label_np.any():
                total_lp_max_risk   += lp_max;  n_risk   += 1
            else:
                total_lp_max_norisk += lp_max;  n_norisk += 1

        valid_idx = gt_valid.nonzero(as_tuple=True)[0]

        # ── Trajectory loss ───────────────────────────────────────────────────
        traj_losses = torch.stack([
            F.l1_loss(pred_traj[k][valid_idx], gt_traj[valid_idx])
            for k in range(K)
        ])  # [K]

        if loss_mode == "laplace":
            log_liks = -traj_losses / laplace_b
            L_traj = math.log(K) - torch.logsumexp(log_liks, dim=0)
        else:
            with torch.no_grad():
                best_k = int(traj_losses.argmin())
            wta_weights = torch.full((K,), 0.3, device=device)
            wta_weights[best_k] = 1.0
            L_traj = (traj_losses * wta_weights).sum()

        # ── L_risk (BCE) — use_risk_prefix=False 시만 ────────────────────────
        if not use_risk_prefix and risk_logits is not None:
            L_risk = bce(risk_logits[0], risk_gt[0])
            loss = L_traj + lam * L_risk
        else:
            loss = L_traj

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

    # lane_prob 요약: risk 있을 때 max_lane_prob이 높을수록 decisive한 차선 선택
    if n_risk > 0 or n_norisk > 0:
        mode_str = "train" if train else "eval"
        lp_risk_str   = f"{total_lp_max_risk/n_risk:.3f}"   if n_risk   else "N/A"
        lp_norisk_str = f"{total_lp_max_norisk/n_norisk:.3f}" if n_norisk else "N/A"
        print(f"  [{mode_str}] lane_prob max  risk={lp_risk_str}  no-risk={lp_norisk_str}  "
              f"(risk scenarios: {n_risk}/{n_ok})")

    return (total_loss / n_ok, total_ade / n_ok,
            total_fde / n_ok, total_mr / n_ok, n_ok)


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Train  : training   shards 00000-00049  ({len(TRAIN_PATHS)} files)")
    print(f"Eval   : validation shards 00000-00007  ({len(EVAL_PATHS)} files)")
    print(f"Epochs : {args.epochs}  lr={args.lr}  wd={args.wd}  λ={args.lam}")
    loss_desc      = f"Laplace NLL (b={args.laplace_b})" if args.loss == "laplace" else "soft-WTA (non-best=0.3)"
    use_lane_mamba  = not args.no_lane_mamba
    use_risk_prefix = not args.no_risk_prefix
    use_traj_fix    = not args.no_traj_fix
    ckpt_dir = os.path.join(CKPT_DIR, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run        : {args.run_name}  →  {ckpt_dir}/")
    print(f"Loss       : {loss_desc}")
    print(f"LaneMamba  : {use_lane_mamba}")
    print(f"RiskPrefix : {use_risk_prefix}")
    print(f"TrajGPT    : {args.use_traj_gpt}")
    print(f"TrajFix    : {use_traj_fix}")
    print()

    if args.use_traj_gpt:
        from src.models.traj_gpt import TrajGPT
        decoder_type = 'gru' if args.gru else 'transformer'
        model = TrajGPT(d_model=128, K=6, n_layers=4, n_heads=4, enc_layers=2,
                        decoder_type=decoder_type).to(device)
    else:
        model = RiskConditionedModel(d_model=128, K=6, n_layers=2,
                                     use_lane_mamba=use_lane_mamba,
                                     use_risk_prefix=use_risk_prefix,
                                     use_traj_fix=use_traj_fix,
                                     use_map_per_step=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt_state = ckpt["optimizer"]
            # move step tensors to CPU to avoid AdamW capturable=False assertion
            for state in opt_state.get("state", {}).values():
                if "step" in state and isinstance(state["step"], torch.Tensor):
                    state["step"] = state["step"].cpu()
            optimizer.load_state_dict(opt_state)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}  (epoch {start_epoch})")

    best_eval_ade = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        # scheduled sampling: epoch 1 → tf=1.0, 마지막 epoch → tf=0.2
        tf_prob = 1.0 - 0.8 * (epoch - 1) / max(args.epochs - 1, 1)

        print(f"{'='*65}")
        print(f"Epoch {epoch}/{args.epochs}  tf_prob={tf_prob:.2f}")
        print(f"{'='*65}")

        if args.use_cache:
            train_npz = sorted(glob.glob(os.path.join(args.cache_dir, "train", "*.npz")))
            val_npz   = sorted(glob.glob(os.path.join(args.cache_dir, "val",   "*.npz")))
            if not train_npz:
                raise FileNotFoundError(f"캐시 없음: {args.cache_dir}/train/  →  먼저 preprocess.py 실행")
            epoch_fn = run_one_epoch_cache
        else:
            train_npz, val_npz = TRAIN_PATHS, EVAL_PATHS
            epoch_fn = lambda model, opt, paths, device, train, **kw: run_one_epoch(
                model, opt, paths, device, train, lam=args.lam, **kw
            )

        model.train()
        tr_loss, tr_ade, tr_fde, tr_mr, n_tr = epoch_fn(
            model, optimizer, train_npz, device, train=True,
            log_every=args.log_every,
            max_scenarios=args.max_scenarios, tf_prob=tf_prob,
            loss_mode=args.loss, laplace_b=args.laplace_b,
            use_risk_prefix=use_risk_prefix,
        )
        print(f"  [train] DONE  loss={tr_loss:.4f}  "
              f"minADE={tr_ade:.3f}m  minFDE={tr_fde:.3f}m  MR={tr_mr:.3f}  "
              f"({n_tr} scenarios)")

        model.eval()
        with torch.no_grad():
            ev_loss, ev_ade, ev_fde, ev_mr, n_ev = epoch_fn(
                model, None, val_npz, device, train=False,
                log_every=args.log_every,
                max_scenarios=args.max_scenarios,
                loss_mode=args.loss, laplace_b=args.laplace_b,
                use_risk_prefix=use_risk_prefix,
            )
        print(f"  [eval]  DONE  loss={ev_loss:.4f}  "
              f"minADE={ev_ade:.3f}m  minFDE={ev_fde:.3f}m  MR={ev_mr:.3f}  "
              f"({n_ev} scenarios)")

        scheduler.step()

        ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch:02d}.pt")
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
            best_path = os.path.join(ckpt_dir, "model_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ev_ade": ev_ade, "ev_fde": ev_fde, "ev_mr": ev_mr},
                       best_path)
            print(f"  ** New best  minADE={ev_ade:.3f}m  "
                  f"minFDE={ev_fde:.3f}m  MR={ev_mr:.3f}  → {best_path}")

        print()

    print(f"학습 완료.  Best eval minADE = {best_eval_ade:.3f}m")
    print(f"Best checkpoint : {os.path.join(ckpt_dir, 'model_best.pt')}")


if __name__ == "__main__":
    main()
