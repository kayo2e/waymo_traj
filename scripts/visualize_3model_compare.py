"""
RCM baseline / RCM traj_fix / TrajGPT+RCM 3-모델 궤적 예측 비교 시각화.

동일한 시나리오에서 세 모델의 K=6 예측을 나란히 표시.
  - RCM baseline : 파란 계열
  - RCM traj_fix : 빨간 계열
  - TrajGPT+RCM  : 초록 계열

Run:
    python visualize_3model_compare.py
    python visualize_3model_compare.py --scenario_id 867742bb79b60ef1
    python visualize_3model_compare.py --curve
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.tfrecord       import iter_tfrecords
from src.data.features       import extract_features, extract_risk_label
from src.models.motion_model import RiskConditionedModel
from src.models.traj_gpt     import TrajGPT
from src.eval.metrics        import compute_minADE_FDE

_VAL_PATHS = [
    os.path.join(ROOT, "waymo-motion-v1_3_0", "val",
                 f"validation.tfrecord-{i:05d}-of-00150")
    for i in range(4)
]

# baseline: 파란 계열
COLOR_BASE_BEST   = "#2166ac"
COLORS_BASE_OTHER = ["#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"]

# traj_fix: 빨간 계열
COLOR_FIX_BEST    = "#cb181d"
COLORS_FIX_OTHER  = ["#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2", "#fff5f0"]

# TrajGPT+RCM: 초록 계열
COLOR_GPT_BEST    = "#1a9641"
COLORS_GPT_OTHER  = ["#74c476", "#a1d99b", "#c7e9c0", "#e5f5e0", "#f7fcf5"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_base",     type=str,
                   default=os.path.join(ROOT, "checkpoints/baseline_quick/model_best.pt"))
    p.add_argument("--ckpt_fix",      type=str,
                   default=os.path.join(ROOT, "checkpoints/traj_fix/model_best.pt"))
    p.add_argument("--ckpt_gpt",      type=str,
                   default=os.path.join(ROOT, "checkpoints/traj_gpt_rcm/model_best.pt"))
    p.add_argument("--scenario_idx",  type=int, default=0)
    p.add_argument("--scenario_id",   type=str, default=None)
    p.add_argument("--curve",         action="store_true",
                   help="커브 시나리오 자동 탐색 (max |lateral| > 12 m)")
    p.add_argument("--curve_thresh",  type=float, default=12.0)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",       type=str, default=".")
    return p.parse_args()


def load_rcm(ckpt_path, use_traj_fix, device):
    model = RiskConditionedModel(
        d_model=128, K=6, n_layers=2,
        use_lane_mamba=False,
        use_risk_prefix=True,
        use_traj_fix=use_traj_fix,
        use_map_per_step=False,
    ).to(device)
    label = "RCM (random)"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep  = ckpt.get("epoch", "?")
        ade = ckpt.get("ev_ade", float("nan"))
        label = f"RCM({'traj_fix' if use_traj_fix else 'base'}) ep{ep}  ADE={ade:.3f}m"
        print(f"[{'traj_fix' if use_traj_fix else 'baseline'}] {label}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}")
    model.eval()
    return model, label


def load_traj_gpt(ckpt_path, device):
    model = TrajGPT(d_model=128, K=6, n_layers=4, n_heads=4, enc_layers=2).to(device)
    label = "TrajGPT+RCM (random)"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep  = ckpt.get("epoch", "?")
        ade = ckpt.get("ev_ade", float("nan"))
        label = f"TrajGPT+RCM ep{ep}  ADE={ade:.3f}m"
        print(f"[traj_gpt_rcm] {label}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}")
    model.eval()
    return model, label


def _prep_inputs(feats, device):
    agent = feats["agent_tensor"]
    scene = feats["scene_tensor"]
    traf  = feats["traffic_tensor"]
    return (
        torch.from_numpy(agent[0:1]).to(device),
        torch.from_numpy(agent[1:][None]).to(device),
        torch.from_numpy(scene[None]).to(device),
        torch.from_numpy(traf[None]).to(device),
    )


def draw_map(ax, scene_tensor, alpha=0.4):
    for poly in scene_tensor:
        xs, ys = poly[:, 0], poly[:, 1]
        if np.all(xs == 0) and np.all(ys == 0):
            continue
        ax.plot(xs, ys, color="lightgray", linewidth=0.9, alpha=alpha, zorder=1)


def add_time_ticks(ax, traj, valid_idx, color, step=10, zorder=6):
    ticks = [i for i in valid_idx if (i + 1) % step == 0]
    if ticks:
        ax.scatter(traj[ticks, 0], traj[ticks, 1],
                   color=color, s=35, zorder=zorder,
                   edgecolors="white", linewidths=0.8)


def add_arrow(ax, traj, valid_idx, color, zorder=7):
    if len(valid_idx) < 2:
        return
    i0, i1 = valid_idx[-2], valid_idx[-1]
    ax.annotate("", xy=(traj[i1, 0], traj[i1, 1]),
                xytext=(traj[i0, 0], traj[i0, 1]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=zorder)


def compute_bounds(arrays, margin=10):
    pts = np.concatenate([a for a in arrays if a is not None])
    return (pts[:, 0].min() - margin, pts[:, 0].max() + margin,
            pts[:, 1].min() - margin, pts[:, 1].max() + margin)


def best_k(pred, gt, vi):
    ades = [np.linalg.norm(pred[k][vi] - gt[vi], axis=1).mean() for k in range(pred.shape[0])]
    return int(np.argmin(ades))


def draw_model(ax, pred, gt, vi, color_best, colors_other, label_best, zorder_base=4):
    bk = best_k(pred, gt, vi)

    for k in range(pred.shape[0]):
        if k == bk:
            continue
        col = colors_other[k % len(colors_other)]
        ax.plot(pred[k][vi, 0], pred[k][vi, 1],
                color=col, linewidth=1.2, linestyle=":", alpha=0.65,
                zorder=zorder_base)

    ax.plot(pred[bk][vi, 0], pred[bk][vi, 1],
            color=color_best, linewidth=2.5, linestyle="-",
            label=label_best, zorder=zorder_base + 2)
    add_time_ticks(ax, pred[bk], vi, color_best, zorder=zorder_base + 3)
    add_arrow(ax, pred[bk], vi, color_best, zorder=zorder_base + 4)
    return bk


def main():
    args   = parse_args()
    device = torch.device(args.device)

    model_base, label_base = load_rcm(args.ckpt_base, use_traj_fix=False, device=device)
    model_fix,  label_fix  = load_rcm(args.ckpt_fix,  use_traj_fix=True,  device=device)
    model_gpt,  label_gpt  = load_traj_gpt(args.ckpt_gpt, device=device)

    from waymo_open_dataset.protos import scenario_pb2

    feats = None
    sc_id = ""
    found = 0
    if args.scenario_id:
        mode_str = f"id={args.scenario_id}"
    elif args.curve:
        mode_str = "curve"
    else:
        mode_str = f"idx={args.scenario_idx}"
    print(f"\n시나리오 탐색 중 ({mode_str})...")

    for raw_bytes in iter_tfrecords(_VAL_PATHS):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            f  = extract_features(sc)
            rl = extract_risk_label(sc)
        except Exception:
            continue
        if not f["gt_valid"].any():
            continue

        if args.scenario_id:
            if sc.scenario_id != args.scenario_id:
                continue
            feats, risk_np, sc_id = f, rl, sc.scenario_id
            print(f"  ID={sc_id} 발견")
            break
        elif args.curve:
            gt, valid = f["gt_trajectory"], f["gt_valid"]
            lat = np.abs(gt[valid, 1]).max() if valid.any() else 0
            if lat < args.curve_thresh:
                found += 1
                continue
            feats, risk_np, sc_id = f, rl, sc.scenario_id
            print(f"  curve found (scanned={found})  max_lat={lat:.1f}m  ID={sc_id}")
            break
        else:
            if found == args.scenario_idx:
                feats, risk_np, sc_id = f, rl, sc.scenario_id
                print(f"  ID={sc_id}")
                break
            found += 1

    if feats is None:
        print("ERROR: 조건에 맞는 시나리오를 찾지 못했습니다.")
        return

    gt_traj  = feats["gt_trajectory"]
    gt_valid = feats["gt_valid"]
    vi       = np.where(gt_valid)[0]

    ego_h, soc, map_s, traf = _prep_inputs(feats, device)
    risk_gt = torch.from_numpy(risk_np).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_base = model_base(ego_h, soc, map_s, traf,
                               risk_label=risk_gt)["trajectory"][0].cpu().numpy()
        pred_fix  = model_fix(ego_h, soc, map_s, traf,
                              risk_label=risk_gt)["trajectory"][0].cpu().numpy()
        pred_gpt  = model_gpt(ego_h, soc, map_s, traf,
                              risk_label=risk_gt)["trajectory"][0].cpu().numpy()

    ade_base, fde_base = compute_minADE_FDE(pred_base, gt_traj, gt_valid)
    ade_fix,  fde_fix  = compute_minADE_FDE(pred_fix,  gt_traj, gt_valid)
    ade_gpt,  fde_gpt  = compute_minADE_FDE(pred_gpt,  gt_traj, gt_valid)

    ego_hist_xy = feats["agent_tensor"][0, :, :2]

    bk_base = best_k(pred_base, gt_traj, vi)
    bk_fix  = best_k(pred_fix,  gt_traj, vi)
    bk_gpt  = best_k(pred_gpt,  gt_traj, vi)

    zx0, zx1, zy0, zy1 = compute_bounds([
        gt_traj[vi], ego_hist_xy,
        pred_base[bk_base][vi], pred_fix[bk_fix][vi], pred_gpt[bk_gpt][vi],
    ], margin=12)

    risk_tags = ["Proximity", "HardBrake", "LaneChange"]
    risk_str  = " | ".join(t for t, v in zip(risk_tags, risk_np) if v) or "No Risk"

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"3-Model Comparison  |  Scenario: {sc_id}  |  Risk: {risk_str}\n"
        f"ego-relative coords  (T=11 hist → 80 future @ 10 Hz = 8 s)",
        fontsize=12, y=1.01
    )

    for ax, is_zoom in [(ax_full, False), (ax_zoom, True)]:
        draw_map(ax, feats["scene_tensor"], alpha=0.45 if is_zoom else 0.3)

        agent = feats["agent_tensor"]
        for i in range(1, agent.shape[0]):
            cx, cy = agent[i, -1, 0], agent[i, -1, 1]
            if cx == 0 and cy == 0:
                continue
            ax.scatter(cx, cy, color="gray",
                       s=20 if is_zoom else 12, alpha=0.4, zorder=3)

        ax.plot(ego_hist_xy[:, 0], ego_hist_xy[:, 1],
                color="black", linewidth=2.5, solid_capstyle="round",
                label="Ego history", zorder=9)
        ax.scatter(ego_hist_xy[-1, 0], ego_hist_xy[-1, 1],
                   color="black", s=90, zorder=10)

        ax.plot(gt_traj[vi, 0], gt_traj[vi, 1],
                color="#2ca02c", linewidth=2.5, linestyle="--",
                label="GT future", zorder=8)
        add_time_ticks(ax, gt_traj, vi, "#2ca02c", zorder=9)
        add_arrow(ax, gt_traj, vi, "#2ca02c", zorder=9)

        draw_model(ax, pred_base, gt_traj, vi,
                   COLOR_BASE_BEST, COLORS_BASE_OTHER,
                   f"RCM base  ADE={ade_base:.2f}m  FDE={fde_base:.2f}m",
                   zorder_base=4)
        draw_model(ax, pred_fix, gt_traj, vi,
                   COLOR_FIX_BEST, COLORS_FIX_OTHER,
                   f"RCM traj_fix★  ADE={ade_fix:.2f}m  FDE={fde_fix:.2f}m",
                   zorder_base=5)
        draw_model(ax, pred_gpt, gt_traj, vi,
                   COLOR_GPT_BEST, COLORS_GPT_OTHER,
                   f"TrajGPT+RCM★★  ADE={ade_gpt:.2f}m  FDE={fde_gpt:.2f}m",
                   zorder_base=6)

        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)  →  ego forward", fontsize=10)
        ax.set_ylabel("y (m)  →  ego left", fontsize=10)

        if is_zoom:
            ax.set_xlim(zx0, zx1)
            ax.set_ylim(zy0, zy1)
            ax.set_title("Zoom: trajectory area", fontsize=11)
            ax.text(0.02, 0.02, "● = 1 s interval (10 steps)",
                    transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
            extra = [
                mpatches.Patch(color="#6baed6", alpha=0.7,
                               label=f"base other {pred_base.shape[0]-1} modes"),
                mpatches.Patch(color="#fb6a4a", alpha=0.7,
                               label=f"traj_fix other {pred_fix.shape[0]-1} modes"),
                mpatches.Patch(color="#74c476", alpha=0.7,
                               label=f"GPT+RCM other {pred_gpt.shape[0]-1} modes"),
            ]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + extra, labels + [p.get_label() for p in extra],
                      loc="upper right", fontsize=9, framealpha=0.92)
        else:
            ax.set_title("Full map view", fontsize=11)
            from matplotlib.patches import Rectangle
            rect = Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                              linewidth=1.5, edgecolor="dimgray",
                              facecolor="none", linestyle="--", zorder=10)
            ax.add_patch(rect)
            ax.text(zx0, zy1 + 2, "zoomed →", fontsize=8, color="dimgray")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"3model_compare_{sc_id}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print(f"  RCM baseline   minADE={ade_base:.3f}m  minFDE={fde_base:.3f}m")
    print(f"  RCM traj_fix   minADE={ade_fix:.3f}m  minFDE={fde_fix:.3f}m")
    print(f"  TrajGPT+RCM    minADE={ade_gpt:.3f}m  minFDE={fde_gpt:.3f}m")


if __name__ == "__main__":
    main()
