"""
Per-maneuver ADE/FDE/MR breakdown evaluation.

Usage:
    conda run -n waymo python eval_maneuver_breakdown.py

각 모델을 val cache 전체로 평가하고
Stop / GoStraight / LaneChangeLeft / LaneChangeRight / TurnLeft / TurnRight
6개 maneuver 클래스별 minADE/FDE/MR 테이블을 출력한다.
"""

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of scripts/)
sys.path.insert(0, ROOT)

from src.data.features       import MAP_DIM
from src.models.motion_model import RiskConditionedModel
from src.eval.metrics        import compute_minADE_FDE, compute_MR

# ── maneuver 클래스 정의 (cond_labels dim 0~5) ────────────────────────────────
MANEUVER_NAMES = ["Stop", "Straight", "LC_Left", "LC_Right", "Turn_Left", "Turn_Right"]
N_MAN = len(MANEUVER_NAMES)

# ── 평가할 모델 목록 ──────────────────────────────────────────────────────────
# (display_name, checkpoint_path, model_kwargs, label_key)
MODELS = [
    (
        "Baseline",
        "checkpoints/causal_maneuver_rare_align/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True),
        "cond_labels",
    ),
    (
        "GoalCond",
        "checkpoints/exp_goal_cond/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_goal_cond=True),
        "cond_labels",
    ),
    (
        "LaneGoal",
        "checkpoints/exp_lane_goal/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_lane_goal=True),
        "cond_labels",
    ),
    (
        "LaneGoal_v2",
        "checkpoints/exp_lane_goal_v2/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_lane_goal=True, use_cond_query=True),
        "cond_labels",
    ),
    (
        "LaneGoal_v3",
        "checkpoints/exp_lane_goal_v3/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_lane_goal=True, use_cond_query=True, use_turn_emb=True),
        "cond_labels",
    ),
    (
        "Combined",
        "checkpoints/exp_goal_lane_combined/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_goal_cond=True, use_lane_goal=True, use_cond_query=True),
        "cond_labels",
    ),
    (
        "LaneGraph",
        "checkpoints/exp_lane_graph/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_goal_cond=True, use_lane_goal=True, use_cond_query=True,
             use_lane_graph=True),
        "cond_labels",
    ),
    (
        "GatedGoal",
        "checkpoints/exp_gated_goal/model_best.pt",
        dict(cond_dim=9, use_lane_mamba=False, use_causal_attn=True,
             use_goal_cond=True, use_lane_goal=True, use_cond_query=True,
             use_goal_gate=True),
        "cond_labels",
    ),
]


def _compute_cond_label(agent, gt_traj, gt_valid):
    label = np.zeros(9, dtype=np.float32)
    vx, vy = agent[0, -1, 2], agent[0, -1, 3]
    v = np.hypot(vx, vy)
    if v < 0.5:   label[6] = 1.0
    elif v < 5.0: label[7] = 1.0
    else:          label[8] = 1.0
    vi = np.where(gt_valid)[0]
    if len(vi) == 0:
        label[0] = 1.0
        return label
    fx, fy = gt_traj[vi[-1]]
    dist = np.hypot(fx, fy)
    if dist < 2.0:            label[0] = 1.0
    elif abs(fy) < 3.0:       label[1] = 1.0
    elif 3.0 <= fy < 10.0:   label[2] = 1.0
    elif -10.0 < fy <= -3.0: label[3] = 1.0
    elif fy >= 10.0:           label[4] = 1.0
    else:                      label[5] = 1.0
    return label


def load_model(ckpt_path, model_kwargs, device):
    defaults = dict(
        d_model=128, K=6, n_layers=2, map_dim=MAP_DIM,
        use_lane_mamba=False, use_risk_prefix=True,
        use_traj_fix=True, use_map_per_step=False,
        use_ar=False, use_lane_anchor=False, cond_dim=9,
        use_causal_attn=False, use_goal_cond=False, use_lane_goal=False,
    )
    defaults.update(model_kwargs)
    model = RiskConditionedModel(**defaults).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    epoch   = ck.get("epoch", "?")
    ev_ade  = ck.get("ev_ade", float("nan"))
    return model, epoch, ev_ade


@torch.no_grad()
def evaluate(model, val_npz_paths, label_key, device):
    """
    Returns:
      overall : dict(ade, fde, mr, n)
      per_man : list of dict(ade, fde, mr, n)  — length N_MAN
    """
    sum_ade = np.zeros(N_MAN + 1)   # last slot = overall
    sum_fde = np.zeros(N_MAN + 1)
    sum_mr  = np.zeros(N_MAN + 1)
    cnt     = np.zeros(N_MAN + 1, dtype=int)

    for path in sorted(val_npz_paths):
        if not os.path.exists(path):
            continue
        data = np.load(path)
        N = len(data["agents"])
        for i in range(N):
            agent      = data["agents"][i]
            scene      = data["scenes"][i]
            traf       = data["trafs"][i]
            gt_traj_np = data["gt_trajs"][i]
            gt_valid_np= data["gt_valids"][i]

            if not gt_valid_np.any():
                continue

            # maneuver class
            cond = data["cond_labels"][i] if "cond_labels" in data \
                   else _compute_cond_label(agent, gt_traj_np, gt_valid_np)
            man_idx = int(cond[:N_MAN].argmax())   # 0~5

            # label for conditioning
            if label_key is None:
                risk_t = None
            elif label_key == "cond_labels":
                risk_t = torch.from_numpy(cond).unsqueeze(0).to(device)
            elif label_key in data:
                risk_t = torch.from_numpy(data[label_key][i]).unsqueeze(0).to(device)
            else:
                risk_t = None

            ego_hist  = torch.from_numpy(agent[0:1]).to(device)
            social    = torch.from_numpy(agent[1:][None]).to(device)
            map_scene = torch.from_numpy(scene[None]).to(device)
            traf_t    = torch.from_numpy(traf[None]).to(device)

            out = model(ego_hist, social, map_scene, traf_t,
                        risk_label=risk_t, gt_traj=None, tf_prob=0.0)
            pred_np = out["trajectory"][0].cpu().numpy()   # [K, 80, 2]

            ade, fde = compute_minADE_FDE(pred_np, gt_traj_np, gt_valid_np)
            mr       = compute_MR(pred_np, gt_traj_np, gt_valid_np)

            if np.isnan(ade):
                continue

            sum_ade[man_idx] += ade;  sum_fde[man_idx] += fde;  sum_mr[man_idx] += mr
            sum_ade[-1]      += ade;  sum_fde[-1]      += fde;  sum_mr[-1]      += mr
            cnt[man_idx]     += 1;    cnt[-1]           += 1

    def _safe(s, c):
        return float(s / c) if c > 0 else float("nan")

    per_man = [
        dict(ade=_safe(sum_ade[k], cnt[k]),
             fde=_safe(sum_fde[k], cnt[k]),
             mr =_safe(sum_mr[k],  cnt[k]),
             n  =int(cnt[k]))
        for k in range(N_MAN)
    ]
    overall = dict(ade=_safe(sum_ade[-1], cnt[-1]),
                   fde=_safe(sum_fde[-1], cnt[-1]),
                   mr =_safe(sum_mr[-1],  cnt[-1]),
                   n  =int(cnt[-1]))
    return overall, per_man


def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "  N/A "


def print_table(results):
    """results: list of (name, epoch, ev_ade_ckpt, overall, per_man)"""
    col_w = 12

    # ── header ───────────────────────────────────────────────────────────────
    header = f"{'Model':<22} {'ep':>3} {'All_ADE':>{col_w}} {'All_MR':>{col_w}}"
    for name in MANEUVER_NAMES:
        header += f" {name+'_ADE':>{col_w}}"
    print(header)
    print("-" * len(header))

    for name, epoch, ade_ckpt, overall, per_man in results:
        row = f"{name:<22} {str(epoch):>3} {fmt(overall['ade']):>{col_w}} {fmt(overall['mr']):>{col_w}}"
        for m in per_man:
            row += f" {fmt(m['ade']):>{col_w}}"
        print(row)

    print()
    # ── count row ────────────────────────────────────────────────────────────
    if results:
        _, _, _, overall, per_man = results[0]
        cnt_row = f"{'# scenarios':<22} {'':>3} {overall['n']:>{col_w}d} {'':>{col_w}}"
        for m in per_man:
            cnt_row += f" {m['n']:>{col_w}d}"
        print(cnt_row)

    print()
    # ── FDE table ─────────────────────────────────────────────────────────────
    print("=== minFDE ===")
    header2 = f"{'Model':<22} {'ep':>3} {'All_FDE':>{col_w}}"
    for name in MANEUVER_NAMES:
        header2 += f" {name+'_FDE':>{col_w}}"
    print(header2)
    print("-" * len(header2))
    for name, epoch, _, overall, per_man in results:
        row = f"{name:<22} {str(epoch):>3} {fmt(overall['fde']):>{col_w}}"
        for m in per_man:
            row += f" {fmt(m['fde']):>{col_w}}"
        print(row)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_npz = sorted(glob.glob(os.path.join(ROOT, "cache", "val", "*.npz")))
    if not val_npz:
        raise FileNotFoundError("cache/val/ 에 .npz 파일 없음")
    print(f"Val shards : {len(val_npz)}  |  device : {device}\n")

    results = []
    for display_name, ckpt_path, model_kwargs, label_key in MODELS:
        full_path = os.path.join(ROOT, ckpt_path)
        if not os.path.exists(full_path):
            print(f"[SKIP] {display_name}  —  {ckpt_path} not found")
            continue

        print(f"Evaluating {display_name} ...", flush=True)
        model, epoch, ev_ade_ckpt = load_model(full_path, model_kwargs, device)
        overall, per_man = evaluate(model, val_npz, label_key, device)
        results.append((display_name, epoch, ev_ade_ckpt, overall, per_man))
        print(f"  done  overall ADE={fmt(overall['ade'])}  MR={fmt(overall['mr'])}")

    print("\n" + "=" * 120)
    print("=== Per-Maneuver Breakdown (minADE, m) ===")
    print("=" * 120)
    print_table(results)


if __name__ == "__main__":
    main()
