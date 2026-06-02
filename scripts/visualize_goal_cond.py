"""
Baseline (causal_maneuver_rare_align) vs GoalCond (exp_goal_cond) 궤적 비교.

val cache NPZ에서 시나리오를 골라 두 모델의 예측을 겹쳐 시각화.
  - Baseline : 파란 계열
  - GoalCond : 붉은 계열

Run:
    cd waymo_traj
    conda run -n waymo python scripts/visualize_goal_cond.py
    conda run -n waymo python scripts/visualize_goal_cond.py --maneuver 5  # TurnRight
    conda run -n waymo python scripts/visualize_goal_cond.py --scenario_idx 3
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.features       import MAP_DIM
from src.models.motion_model import RiskConditionedModel
from src.eval.metrics        import compute_minADE_FDE

MANEUVER_NAMES = ["Stop", "Straight", "LC_Left", "LC_Right", "Turn_Left", "Turn_Right"]

# Baseline: 파란 계열
COLOR_BASE_BEST   = "#2166ac"
COLORS_BASE_OTHER = ["#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"]

# GoalCond: 붉은 계열
COLOR_GOAL_BEST   = "#cb181d"
COLORS_GOAL_OTHER = ["#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2", "#fff5f0"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_base", type=str,
                   default=os.path.join(ROOT, "checkpoints/causal_maneuver_rare_align/model_best.pt"))
    p.add_argument("--ckpt_goal", type=str,
                   default=os.path.join(ROOT, "checkpoints/exp_goal_cond/model_best.pt"))
    p.add_argument("--maneuver",     type=int, default=5,
                   help="0=Stop 1=Straight 2=LC_Left 3=LC_Right 4=Turn_Left 5=Turn_Right")
    p.add_argument("--scenario_idx", type=int, default=0,
                   help="해당 maneuver 클래스 내 몇 번째 시나리오 (0-based)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default=ROOT)
    return p.parse_args()


def load_model(ckpt_path, use_goal_cond, device):
    model = RiskConditionedModel(
        d_model=128, K=6, n_layers=2, n_heads=4,
        map_dim=MAP_DIM,
        use_lane_mamba=False,
        use_risk_prefix=True,
        use_traj_fix=True,
        use_causal_attn=True,
        use_goal_cond=use_goal_cond,
        cond_dim=9,
    ).to(device)
    label = "random init"
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep    = ckpt.get("epoch", "?")
        ade   = ckpt.get("ev_ade", float("nan"))
        label = f"ep{ep}  ADE={ade:.3f}m"
        print(f"[{'GoalCond' if use_goal_cond else 'Baseline'}] {ckpt_path}  →  {label}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}")
    model.eval()
    return model, label


def draw_map(ax, scene, alpha=0.4):
    for poly in scene:
        xs, ys = poly[:, 0], poly[:, 1]
        if np.all(xs == 0) and np.all(ys == 0):
            continue
        ax.plot(xs, ys, color="lightgray", linewidth=0.9, alpha=alpha, zorder=1)


def best_k(pred, gt, vi):
    return int(np.argmin([
        np.linalg.norm(pred[k][vi] - gt[vi], axis=1).mean()
        for k in range(pred.shape[0])
    ]))


def add_time_ticks(ax, traj, vi, color, step=10, zorder=6):
    ticks = [i for i in vi if (i + 1) % step == 0]
    if ticks:
        ax.scatter(traj[ticks, 0], traj[ticks, 1],
                   color=color, s=35, zorder=zorder,
                   edgecolors="white", linewidths=0.8)


def add_arrow(ax, traj, vi, color, zorder=7):
    if len(vi) < 2:
        return
    i0, i1 = vi[-2], vi[-1]
    ax.annotate("", xy=(traj[i1, 0], traj[i1, 1]),
                xytext=(traj[i0, 0], traj[i0, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=14),
                zorder=zorder)


def draw_model(ax, pred, gt, vi, color_best, colors_other, label_best, zorder_base=4):
    bk = best_k(pred, gt, vi)
    for k in range(pred.shape[0]):
        if k == bk:
            continue
        ax.plot(pred[k][vi, 0], pred[k][vi, 1],
                color=colors_other[k % len(colors_other)],
                linewidth=1.2, linestyle=":", alpha=0.65, zorder=zorder_base)
    ax.plot(pred[bk][vi, 0], pred[bk][vi, 1],
            color=color_best, linewidth=2.5, linestyle="-",
            label=label_best, zorder=zorder_base + 2)
    add_time_ticks(ax, pred[bk], vi, color_best, zorder=zorder_base + 3)
    add_arrow(ax, pred[bk], vi, color_best, zorder=zorder_base + 4)
    return bk


def compute_bounds(arrays, margin=12):
    pts = np.concatenate([a for a in arrays if a is not None and len(a) > 0])
    return (pts[:, 0].min() - margin, pts[:, 0].max() + margin,
            pts[:, 1].min() - margin, pts[:, 1].max() + margin)


def main():
    args   = parse_args()
    device = torch.device(args.device)

    model_base, label_base = load_model(args.ckpt_base, use_goal_cond=False, device=device)
    model_goal, label_goal = load_model(args.ckpt_goal, use_goal_cond=True,  device=device)

    # ── val cache 스캔 ────────────────────────────────────────────────────────
    val_npz = sorted(glob.glob(os.path.join(ROOT, "cache", "val", "*.npz")))
    if not val_npz:
        raise FileNotFoundError("cache/val/*.npz 없음")
    print(f"\nVal shards: {len(val_npz)}  |  찾는 maneuver: {MANEUVER_NAMES[args.maneuver]}\n")

    found_cnt = 0
    selected  = None
    sc_idx_global = 0

    for npz_path in val_npz:
        data = np.load(npz_path)
        N    = len(data["agents"])
        for i in range(N):
            gt_valid = data["gt_valids"][i]
            if not gt_valid.any():
                continue

            cond    = data["cond_labels"][i]
            man_idx = int(cond[:6].argmax())
            if man_idx != args.maneuver:
                continue

            if found_cnt == args.scenario_idx:
                selected = {
                    "agent":    data["agents"][i],
                    "scene":    data["scenes"][i],
                    "traf":     data["trafs"][i],
                    "gt_traj":  data["gt_trajs"][i],
                    "gt_valid": gt_valid,
                    "cond":     cond,
                    "sc_global_idx": sc_idx_global,
                }
                print(f"시나리오 선택 (global_idx={sc_idx_global}, "
                      f"maneuver={MANEUVER_NAMES[man_idx]})")
                break
            found_cnt += 1
            sc_idx_global += 1
        if selected is not None:
            break
        sc_idx_global += N

    if selected is None:
        print(f"ERROR: maneuver={MANEUVER_NAMES[args.maneuver]} 시나리오를 "
              f"{args.scenario_idx}번 찾지 못했습니다.")
        return

    agent    = selected["agent"]
    scene    = selected["scene"]
    traf     = selected["traf"]
    gt_traj  = selected["gt_traj"]
    gt_valid = selected["gt_valid"]
    cond     = selected["cond"]
    vi       = np.where(gt_valid)[0]

    # ── 추론 ──────────────────────────────────────────────────────────────────
    ego_h  = torch.from_numpy(agent[0:1]).to(device)
    soc    = torch.from_numpy(agent[1:][None]).to(device)
    map_s  = torch.from_numpy(scene[None]).to(device)
    traf_t = torch.from_numpy(traf[None]).to(device)
    cond_t = torch.from_numpy(cond).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_base = model_base(ego_h, soc, map_s, traf_t,
                               risk_label=cond_t, gt_traj=None, tf_prob=0.0
                               )["trajectory"][0].cpu().numpy()
        pred_goal = model_goal(ego_h, soc, map_s, traf_t,
                               risk_label=cond_t, gt_traj=None, tf_prob=0.0
                               )["trajectory"][0].cpu().numpy()

    ade_base, fde_base = compute_minADE_FDE(pred_base, gt_traj, gt_valid)
    ade_goal, fde_goal = compute_minADE_FDE(pred_goal, gt_traj, gt_valid)

    ego_hist_xy = agent[0, :, :2]
    bk_base = best_k(pred_base, gt_traj, vi)
    bk_goal = best_k(pred_goal, gt_traj, vi)

    # speed label
    vx, vy = agent[0, -1, 2], agent[0, -1, 3]
    v = np.hypot(vx, vy)
    speed_tag = "Slow" if v < 5.0 else ("Medium" if v < 10.0 else "Fast")
    man_tag   = MANEUVER_NAMES[args.maneuver]

    zx0, zx1, zy0, zy1 = compute_bounds([
        gt_traj[vi], ego_hist_xy,
        pred_base[bk_base][vi], pred_goal[bk_goal][vi],
    ], margin=12)

    # ── 캔버스 ────────────────────────────────────────────────────────────────
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(20, 8))
    sc_tag = f"idx={selected['sc_global_idx']}"
    fig.suptitle(
        f"Ablation: Baseline / GoalCond  |  Scenario: {sc_tag}  |  Maneuver: {man_tag}/{speed_tag}\n"
        f"ego-relative coords  (T=11 hist → 80 future @ 10 Hz = 8 s)",
        fontsize=12, y=1.01
    )

    for ax, is_zoom in [(ax_full, False), (ax_zoom, True)]:
        draw_map(ax, scene, alpha=0.45 if is_zoom else 0.3)

        # 주변 에이전트
        for i in range(1, agent.shape[0]):
            cx, cy = agent[i, -1, 0], agent[i, -1, 1]
            if cx == 0 and cy == 0:
                continue
            ax.scatter(cx, cy, color="gray", s=20 if is_zoom else 12,
                       alpha=0.4, zorder=3)

        # 에고 히스토리
        ax.plot(ego_hist_xy[:, 0], ego_hist_xy[:, 1],
                color="black", linewidth=2.5, solid_capstyle="round",
                label="Ego history", zorder=9)
        ax.scatter(ego_hist_xy[-1, 0], ego_hist_xy[-1, 1],
                   color="black", s=90, zorder=10)

        # GT future
        ax.plot(gt_traj[vi, 0], gt_traj[vi, 1],
                color="#2ca02c", linewidth=2.5, linestyle="--",
                label="GT future", zorder=8)
        add_time_ticks(ax, gt_traj, vi, "#2ca02c", zorder=9)
        add_arrow(ax, gt_traj, vi, "#2ca02c", zorder=9)

        # Baseline
        draw_model(ax, pred_base, gt_traj, vi,
                   COLOR_BASE_BEST, COLORS_BASE_OTHER,
                   f"Baseline  ADE={ade_base:.2f}m  FDE={fde_base:.2f}m",
                   zorder_base=4)

        # GoalCond
        draw_model(ax, pred_goal, gt_traj, vi,
                   COLOR_GOAL_BEST, COLORS_GOAL_OTHER,
                   f"GoalCond★  ADE={ade_goal:.2f}m  FDE={fde_goal:.2f}m",
                   zorder_base=5)

        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)  →  ego forward", fontsize=10)
        ax.set_ylabel("y (m)  →  ego left",    fontsize=10)

        if is_zoom:
            ax.set_xlim(zx0, zx1)
            ax.set_ylim(zy0, zy1)
            ax.set_title("Zoom: trajectory area", fontsize=11)
            ax.text(0.02, 0.02, "● = 1 s interval (10 steps)",
                    transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
            extra = [
                mpatches.Patch(color="#6baed6", alpha=0.7,
                               label=f"Baseline other {pred_base.shape[0]-1} modes"),
                mpatches.Patch(color="#fb6a4a", alpha=0.7,
                               label=f"GoalCond other {pred_goal.shape[0]-1} modes"),
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

    import hashlib, time
    h   = hashlib.md5(f"{sc_tag}{time.time()}".encode()).hexdigest()[:16]
    out = os.path.join(args.out_dir, f"ablation_goal_cond_{h}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out}")
    print(f"  Baseline  minADE={ade_base:.3f}m  minFDE={fde_base:.3f}m")
    print(f"  GoalCond  minADE={ade_goal:.3f}m  minFDE={fde_goal:.3f}m")
    delta_ade = ade_base - ade_goal
    delta_fde = fde_base - fde_goal
    print(f"  ΔADE={delta_ade:+.3f}m  ΔFDE={delta_fde:+.3f}m  "
          f"({'GoalCond 우세 ✓' if delta_ade > 0 else 'Baseline 우세'})")


if __name__ == "__main__":
    main()
