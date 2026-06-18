"""
LaneGraph (exp_lane_graph) 궤적 시각화.

Baseline vs LaneGraph 비교 또는 LaneGraph 단독.
각 maneuver 클래스 대표 시나리오를 그리드로 출력.

Run:
    conda run -n waymo python scripts/visualize_lane_graph.py
    conda run -n waymo python scripts/visualize_lane_graph.py --maneuver 4   # Turn_Left
    conda run -n waymo python scripts/visualize_lane_graph.py --n_scenarios 3
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

COLOR_BASE_BEST   = "#2166ac"
COLORS_BASE_OTHER = ["#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"]
COLOR_LG_BEST     = "#e91e63"
COLORS_LG_OTHER   = ["#f06292", "#f48fb1", "#f8bbd0", "#fce4ec", "#fff0f3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maneuver", type=int, default=None,
                   help="0=Stop 1=Straight 2=LC_Left 3=LC_Right 4=Turn_Left 5=Turn_Right "
                        "(None = 모든 클래스 6종 그리드)")
    p.add_argument("--scenario_idx", type=int, default=0)
    p.add_argument("--n_scenarios",  type=int, default=1,
                   help="maneuver 지정 시 연속 몇 개 그릴지")
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=ROOT)
    return p.parse_args()


def load_model(ckpt_path, kwargs, device):
    defaults = dict(d_model=128, K=6, n_layers=2, n_heads=4, map_dim=MAP_DIM,
                    use_lane_mamba=False, use_risk_prefix=True,
                    use_traj_fix=True, use_causal_attn=True, cond_dim=9)
    defaults.update(kwargs)
    m = RiskConditionedModel(**defaults).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(ck["model"])
    m.eval()
    ep  = ck.get("epoch", "?")
    ade = ck.get("ev_ade", float("nan"))
    print(f"  loaded {os.path.basename(os.path.dirname(ckpt_path))}  ep{ep}  ADE={ade:.3f}")
    return m, ep, ade


def draw_map(ax, scene, alpha=0.45):
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


def add_ticks(ax, traj, vi, color, step=10, zorder=6):
    ticks = [i for i in vi if (i + 1) % step == 0]
    if ticks:
        ax.scatter(traj[ticks, 0], traj[ticks, 1],
                   color=color, s=30, zorder=zorder,
                   edgecolors="white", linewidths=0.7)


def add_arrow(ax, traj, vi, color, zorder=7):
    if len(vi) < 2:
        return
    i0, i1 = vi[-2], vi[-1]
    ax.annotate("", xy=(traj[i1, 0], traj[i1, 1]),
                xytext=(traj[i0, 0], traj[i0, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, mutation_scale=12),
                zorder=zorder)


def draw_pred(ax, pred, gt, vi, color_best, colors_other, label, zorder_base=4):
    bk = best_k(pred, gt, vi)
    for k in range(pred.shape[0]):
        if k == bk:
            continue
        ax.plot(pred[k][vi, 0], pred[k][vi, 1],
                color=colors_other[k % len(colors_other)],
                linewidth=1.0, linestyle=":", alpha=0.6, zorder=zorder_base)
    ax.plot(pred[bk][vi, 0], pred[bk][vi, 1],
            color=color_best, linewidth=2.3, label=label, zorder=zorder_base + 2)
    add_ticks(ax, pred[bk], vi, color_best, zorder=zorder_base + 3)
    add_arrow(ax, pred[bk], vi, color_best, zorder=zorder_base + 4)
    return bk


def collect_scenarios(val_npz, maneuver, n, start_idx=0):
    results = []
    found = 0
    for path in val_npz:
        data = np.load(path)
        N = len(data["agents"])
        for i in range(N):
            gv = data["gt_valids"][i]
            if not gv.any():
                continue
            cond = data["cond_labels"][i]
            mid  = int(cond[:6].argmax())
            if mid != maneuver:
                continue
            if found < start_idx:
                found += 1
                continue
            results.append({
                "agent":    data["agents"][i],
                "scene":    data["scenes"][i],
                "traf":     data["trafs"][i],
                "gt_traj":  data["gt_trajs"][i],
                "gt_valid": gv,
                "cond":     cond,
                "maneuver": mid,
            })
            found += 1
            if len(results) == n:
                return results
    return results


def render_scenario(ax, sc, model_base, model_lg, device, show_baseline):
    agent    = sc["agent"]
    scene    = sc["scene"]
    traf     = sc["traf"]
    gt_traj  = sc["gt_traj"]
    gt_valid = sc["gt_valid"]
    cond     = sc["cond"]
    vi       = np.where(gt_valid)[0]

    ego_h  = torch.from_numpy(agent[0:1]).to(device)
    soc    = torch.from_numpy(agent[1:][None]).to(device)
    map_s  = torch.from_numpy(scene[None]).to(device)
    traf_t = torch.from_numpy(traf[None]).to(device)
    cond_t = torch.from_numpy(cond).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_lg = model_lg(ego_h, soc, map_s, traf_t,
                           risk_label=cond_t, gt_traj=None, tf_prob=0.0
                           )["trajectory"][0].cpu().numpy()
        if show_baseline:
            pred_base = model_base(ego_h, soc, map_s, traf_t,
                                   risk_label=cond_t, gt_traj=None, tf_prob=0.0
                                   )["trajectory"][0].cpu().numpy()

    ade_lg, fde_lg = compute_minADE_FDE(pred_lg, gt_traj, gt_valid)
    if show_baseline:
        ade_base, fde_base = compute_minADE_FDE(pred_base, gt_traj, gt_valid)

    draw_map(ax, scene)

    # social agents
    for j in range(1, agent.shape[0]):
        cx, cy = agent[j, -1, 0], agent[j, -1, 1]
        if cx == 0 and cy == 0:
            continue
        ax.scatter(cx, cy, color="gray", s=15, alpha=0.35, zorder=2)

    ego_xy = agent[0, :, :2]
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="black", linewidth=2.0,
            solid_capstyle="round", label="Ego hist", zorder=9)
    ax.scatter(ego_xy[-1, 0], ego_xy[-1, 1], color="black", s=70, zorder=10)

    ax.plot(gt_traj[vi, 0], gt_traj[vi, 1], color="#2ca02c", linewidth=2.2,
            linestyle="--", label="GT", zorder=8)
    add_ticks(ax, gt_traj, vi, "#2ca02c", zorder=9)
    add_arrow(ax, gt_traj, vi, "#2ca02c", zorder=9)

    if show_baseline:
        draw_pred(ax, pred_base, gt_traj, vi,
                  COLOR_BASE_BEST, COLORS_BASE_OTHER,
                  f"Baseline  {ade_base:.2f}m", zorder_base=4)

    draw_pred(ax, pred_lg, gt_traj, vi,
              COLOR_LG_BEST, COLORS_LG_OTHER,
              f"LaneGraph  {ade_lg:.2f}m", zorder_base=6)

    # 뷰 범위
    bk_lg = best_k(pred_lg, gt_traj, vi)
    pts = np.concatenate([gt_traj[vi], ego_xy, pred_lg[bk_lg][vi]])
    mg  = 14
    ax.set_xlim(pts[:, 0].min() - mg, pts[:, 0].max() + mg)
    ax.set_ylim(pts[:, 1].min() - mg, pts[:, 1].max() + mg)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.18)

    man_name = MANEUVER_NAMES[sc["maneuver"]]
    vx, vy   = agent[0, -1, 2], agent[0, -1, 3]
    speed    = np.hypot(vx, vy)
    spd_tag  = "Slow" if speed < 5 else ("Med" if speed < 10 else "Fast")
    ax.set_title(f"{man_name} / {spd_tag}", fontsize=10, pad=4)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print("Loading models...")
    model_lg, ep_lg, ade_lg = load_model(
        os.path.join(ROOT, "checkpoints/exp_lane_graph/model_best.pt"),
        dict(use_goal_cond=True, use_lane_goal=True,
             use_cond_query=True, use_lane_graph=True),
        device,
    )
    show_baseline = not args.baseline_only
    if show_baseline:
        model_base, ep_base, ade_base = load_model(
            os.path.join(ROOT, "checkpoints/causal_maneuver_rare_align/model_best.pt"),
            dict(),
            device,
        )
    else:
        model_base = None

    val_npz = sorted(glob.glob(os.path.join(ROOT, "cache", "val", "*.npz")))
    if not val_npz:
        raise FileNotFoundError("cache/val/*.npz 없음")

    # ── 단일 maneuver 모드 ────────────────────────────────────────────────────
    if args.maneuver is not None:
        scenarios = collect_scenarios(val_npz, args.maneuver,
                                      args.n_scenarios, args.scenario_idx)
        if not scenarios:
            print(f"No scenarios for maneuver={MANEUVER_NAMES[args.maneuver]}")
            return
        n = len(scenarios)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
        if n == 1:
            axes = [axes]
        for ax, sc in zip(axes, scenarios):
            render_scenario(ax, sc, model_base, model_lg, device, show_baseline)

        title = (f"LaneGraph ep{ep_lg} (ADE={ade_lg:.3f}m) vs Baseline"
                 if show_baseline else f"LaneGraph ep{ep_lg}")
        fig.suptitle(title, fontsize=12, y=1.02)

    # ── 전체 maneuver 그리드 (6종) ───────────────────────────────────────────
    else:
        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        axes = axes.flatten()
        for man_idx, ax in enumerate(axes):
            scs = collect_scenarios(val_npz, man_idx, 1, args.scenario_idx)
            if not scs:
                ax.set_visible(False)
                continue
            render_scenario(ax, scs[0], model_base, model_lg, device, show_baseline)

        patches = [
            mpatches.Patch(color=COLOR_BASE_BEST, label=f"Baseline (best={ade_base:.3f}m)" if show_baseline else ""),
            mpatches.Patch(color=COLOR_LG_BEST,   label=f"LaneGraph ep{ep_lg} (best={ade_lg:.3f}m)"),
            mpatches.Patch(color="#2ca02c",        label="Ground Truth"),
        ]
        fig.legend(handles=[p for p in patches if p.get_label()],
                   loc="lower center", ncol=3, fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(
            f"LaneGraph ep{ep_lg} (ADE={ade_lg:.3f}m) — All Maneuvers\n"
            "Baseline(blue) vs LaneGraph(pink) | GT(green--) | dots = 1 s intervals",
            fontsize=12, y=1.02,
        )

    plt.tight_layout()
    suffix = f"man{args.maneuver}" if args.maneuver is not None else "all"
    out = os.path.join(args.out_dir, f"viz_lane_graph_{suffix}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
