"""
Maneuver_base / Man+Div / Man+RareAlign 세 모델 궤적 예측 비교 시각화.
val cache NPZ 에서 Turn 시나리오(기본)를 자동 탐색해 풀맵+줌 2-패널 그림 저장.

Usage:
    conda run -n waymo python scripts/viz_maneuver_ablation.py
    conda run -n waymo python scripts/viz_maneuver_ablation.py --maneuver turn --n_samples 6
    conda run -n waymo python scripts/viz_maneuver_ablation.py --maneuver lc --shard shard_00002.npz --idx 5
    conda run -n waymo python scripts/viz_maneuver_ablation.py --shard shard_00000.npz --idx 51
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

# ── 색상 정의 ─────────────────────────────────────────────────────────────────
# Maneuver_base : 회색
COLOR_BASE_BEST   = "#636363"
COLORS_BASE_OTHER = ["#969696", "#bdbdbd", "#d9d9d9", "#f0f0f0", "#ffffff"]

# Man+Div : 파란 계열
COLOR_DIV_BEST   = "#2166ac"
COLORS_DIV_OTHER = ["#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"]

# Man+RareAlign (best) : 빨간/주황 계열
COLOR_RARE_BEST   = "#cb181d"
COLORS_RARE_OTHER = ["#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2", "#fff5f0"]

MANEUVER_NAMES = ["Stop", "Straight", "LCLeft", "LCRight", "TurnLeft", "TurnRight"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_base", default="checkpoints/noar_maneuver_nolane_v2/model_best.pt")
    p.add_argument("--ckpt_div",  default="checkpoints/noar_maneuver_div/model_best.pt")
    p.add_argument("--ckpt_rare", default="checkpoints/noar_maneuver_rare_align/model_best.pt")
    p.add_argument("--val_dir",   default="cache/val")
    p.add_argument("--out_dir",   default=".")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    # 시나리오 선택
    p.add_argument("--shard",     default=None,    help="특정 shard 파일명 (e.g. shard_00000.npz)")
    p.add_argument("--idx",       type=int, default=None, help="shard 내 인덱스 (--shard와 함께)")
    p.add_argument("--maneuver",  default="turn",
                   choices=["turn", "lc", "straight", "stop"],
                   help="자동 탐색할 maneuver 유형 (shard/idx 미지정 시 사용)")
    p.add_argument("--n_samples", type=int, default=1, help="저장할 시나리오 수")
    return p.parse_args()


def _compute_cond_label(agent, gt_traj, gt_valid):
    label = np.zeros(9, dtype=np.float32)
    vx, vy = agent[0, -1, 2], agent[0, -1, 3]
    v = np.hypot(vx, vy)
    if v < 0.5:    label[6] = 1.0
    elif v < 5.0:  label[7] = 1.0
    else:           label[8] = 1.0
    vi = np.where(gt_valid)[0]
    if len(vi) == 0:
        label[0] = 1.0
        return label
    fx, fy = gt_traj[vi[-1]]
    dist = np.hypot(fx, fy)
    if dist < 2.0:            label[0] = 1.0
    elif abs(fy) < 3.0:       label[1] = 1.0
    elif 3.0 <= fy < 10.0:    label[2] = 1.0
    elif -10.0 < fy <= -3.0:  label[3] = 1.0
    elif fy >= 10.0:            label[4] = 1.0
    else:                       label[5] = 1.0
    return label


def _maneuver_filter(man_idx, maneuver_arg):
    if maneuver_arg == "turn":    return man_idx in [4, 5]
    if maneuver_arg == "lc":     return man_idx in [2, 3]
    if maneuver_arg == "straight": return man_idx == 1
    if maneuver_arg == "stop":   return man_idx == 0
    return True


def load_model(rel_path, cond_dim, device):
    ckpt_path = os.path.join(ROOT, rel_path)
    model = RiskConditionedModel(
        d_model=128, K=6, n_layers=2, map_dim=MAP_DIM,
        use_lane_mamba=False, use_risk_prefix=True,
        use_traj_fix=True, use_map_per_step=False,
        use_ar=False, use_lane_anchor=False, cond_dim=cond_dim,
    ).to(device)
    label = os.path.basename(os.path.dirname(rel_path))
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        ep  = ck.get("epoch", "?")
        ade = ck.get("ev_ade", float("nan"))
        label = f"{label} ep{ep} ADE={ade:.3f}m"
        print(f"  loaded: {label}")
    else:
        print(f"  [WARN] not found: {ckpt_path}")
    model.eval()
    return model, label


# ── 그리기 유틸 ───────────────────────────────────────────────────────────────
def draw_map(ax, scene, alpha=0.4):
    for poly in scene:
        xs, ys = poly[:, 0], poly[:, 1]
        if np.all(xs == 0) and np.all(ys == 0):
            continue
        ax.plot(xs, ys, color="lightgray", linewidth=0.9, alpha=alpha, zorder=1)


def add_time_ticks(ax, traj, vi, color, step=10, zorder=6):
    ticks = [i for i in vi if (i + 1) % step == 0]
    if ticks:
        ax.scatter(traj[ticks, 0], traj[ticks, 1], color=color, s=35, zorder=zorder,
                   edgecolors="white", linewidths=0.8)


def add_arrow(ax, traj, vi, color, zorder=7):
    if len(vi) < 2:
        return
    i0, i1 = vi[-2], vi[-1]
    ax.annotate("", xy=(traj[i1, 0], traj[i1, 1]), xytext=(traj[i0, 0], traj[i0, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=14),
                zorder=zorder)


def best_k(pred, gt, vi):
    ades = [np.linalg.norm(pred[k][vi] - gt[vi], axis=1).mean() for k in range(pred.shape[0])]
    return int(np.argmin(ades))


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


def visualize_one(data, idx, models_info, device, out_dir):
    """
    models_info: list of (model, label, color_best, colors_other)
    """
    m_base, l_base = models_info[0]
    m_div,  l_div  = models_info[1]
    m_rare, l_rare = models_info[2]

    agent_np    = data["agents"][idx]
    scene_np    = data["scenes"][idx]
    traf_np     = data["trafs"][idx]
    gt_traj_np  = data["gt_trajs"][idx]
    gt_valid_np = data["gt_valids"][idx]
    sc_id       = str(data["sc_ids"][idx])

    cond_np = (data["cond_labels"][idx] if "cond_labels" in data
               else _compute_cond_label(agent_np, gt_traj_np, gt_valid_np))

    vi = np.where(gt_valid_np)[0]
    if len(vi) == 0:
        return None

    def t(x): return torch.from_numpy(x).to(device)
    ego_h  = t(agent_np[0]).unsqueeze(0)       # [1, 11, 10]
    soc    = t(agent_np[1:]).unsqueeze(0)       # [1, 31, 11, 10]
    map_s  = t(scene_np).unsqueeze(0)           # [1, 50, ?, 8]
    traf   = t(traf_np).unsqueeze(0)            # [1, 6, 1]
    cond_t = t(cond_np).unsqueeze(0)            # [1, 9]

    with torch.no_grad():
        pred_base = m_base(ego_h, soc, map_s, traf, risk_label=cond_t)["trajectory"][0].cpu().numpy()
        pred_div  = m_div(ego_h, soc, map_s, traf, risk_label=cond_t)["trajectory"][0].cpu().numpy()
        pred_rare = m_rare(ego_h, soc, map_s, traf, risk_label=cond_t)["trajectory"][0].cpu().numpy()

    ade_base, fde_base = compute_minADE_FDE(pred_base, gt_traj_np, gt_valid_np)
    ade_div,  fde_div  = compute_minADE_FDE(pred_div,  gt_traj_np, gt_valid_np)
    ade_rare, fde_rare = compute_minADE_FDE(pred_rare, gt_traj_np, gt_valid_np)

    ego_hist_xy = agent_np[0, :, :2]
    bk_base = best_k(pred_base, gt_traj_np, vi)
    bk_div  = best_k(pred_div,  gt_traj_np, vi)
    bk_rare = best_k(pred_rare, gt_traj_np, vi)

    zx0, zx1, zy0, zy1 = compute_bounds([
        gt_traj_np[vi], ego_hist_xy,
        pred_base[bk_base][vi], pred_div[bk_div][vi], pred_rare[bk_rare][vi],
    ], margin=12)

    man_idx = int(cond_np[:6].argmax())
    spd_idx = int(cond_np[6:].argmax())
    man_str = f"{MANEUVER_NAMES[man_idx]}/{'Stopped Slow Fast'.split()[spd_idx]}"

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"Ablation: Maneuver_base / Man+Div / Man+RareAlign  |  Scenario: {sc_id}  |  Maneuver: {man_str}\n"
        f"ego-relative coords  (T=11 hist → 80 future @ 10 Hz = 8 s)",
        fontsize=12, y=1.01
    )

    for ax, is_zoom in [(ax_full, False), (ax_zoom, True)]:
        draw_map(ax, scene_np, alpha=0.45 if is_zoom else 0.3)

        for i in range(1, agent_np.shape[0]):
            cx, cy = agent_np[i, -1, 0], agent_np[i, -1, 1]
            if cx == 0 and cy == 0:
                continue
            ax.scatter(cx, cy, color="gray", s=20 if is_zoom else 12, alpha=0.4, zorder=3)

        ax.plot(ego_hist_xy[:, 0], ego_hist_xy[:, 1],
                color="black", linewidth=2.5, solid_capstyle="round",
                label="Ego history", zorder=9)
        ax.scatter(ego_hist_xy[-1, 0], ego_hist_xy[-1, 1],
                   color="black", s=90, zorder=10)

        ax.plot(gt_traj_np[vi, 0], gt_traj_np[vi, 1],
                color="#2ca02c", linewidth=2.5, linestyle="--",
                label="GT future", zorder=8)
        add_time_ticks(ax, gt_traj_np, vi, "#2ca02c", zorder=9)
        add_arrow(ax, gt_traj_np, vi, "#2ca02c", zorder=9)

        draw_model(ax, pred_base, gt_traj_np, vi,
                   COLOR_BASE_BEST, COLORS_BASE_OTHER,
                   f"Maneuver_base  ADE={ade_base:.2f}m  FDE={fde_base:.2f}m",
                   zorder_base=4)
        draw_model(ax, pred_div, gt_traj_np, vi,
                   COLOR_DIV_BEST, COLORS_DIV_OTHER,
                   f"Man+Div  ADE={ade_div:.2f}m  FDE={fde_div:.2f}m",
                   zorder_base=5)
        draw_model(ax, pred_rare, gt_traj_np, vi,
                   COLOR_RARE_BEST, COLORS_RARE_OTHER,
                   f"Man+RareAlign★  ADE={ade_rare:.2f}m  FDE={fde_rare:.2f}m",
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
                mpatches.Patch(color="#969696", alpha=0.7,
                               label=f"Maneuver_base other {pred_base.shape[0]-1} modes"),
                mpatches.Patch(color="#6baed6", alpha=0.7,
                               label=f"Man+Div other {pred_div.shape[0]-1} modes"),
                mpatches.Patch(color="#fb6a4a", alpha=0.7,
                               label=f"Man+RareAlign other {pred_rare.shape[0]-1} modes"),
            ]
            handles, labels_leg = ax.get_legend_handles_labels()
            ax.legend(handles + extra, labels_leg + [p.get_label() for p in extra],
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
    out_path = os.path.join(out_dir, f"ablation_maneuver_{sc_id}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"    Maneuver_base  ADE={ade_base:.3f}m  FDE={fde_base:.3f}m")
    print(f"    Man+Div        ADE={ade_div:.3f}m  FDE={fde_div:.3f}m")
    print(f"    Man+RareAlign  ADE={ade_rare:.3f}m  FDE={fde_rare:.3f}m")
    return out_path


def main():
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = os.path.join(ROOT, args.out_dir)

    print("모델 로드 중...")
    m_base, l_base = load_model(args.ckpt_base, cond_dim=9, device=device)
    m_div,  l_div  = load_model(args.ckpt_div,  cond_dim=9, device=device)
    m_rare, l_rare = load_model(args.ckpt_rare, cond_dim=9, device=device)
    models_info = [(m_base, l_base), (m_div, l_div), (m_rare, l_rare)]

    val_dir = os.path.join(ROOT, args.val_dir)

    # ── 특정 shard+idx 지정 ────────────────────────────────────────────────────
    if args.shard is not None and args.idx is not None:
        shard_path = os.path.join(val_dir, args.shard)
        print(f"\n단일 시나리오: {args.shard} idx={args.idx}")
        data = np.load(shard_path, allow_pickle=True)
        visualize_one(data, args.idx, models_info, device, out_dir)
        return

    # ── maneuver 유형 자동 탐색 ────────────────────────────────────────────────
    npz_paths = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    print(f"\n{args.maneuver} 시나리오 탐색 중 (n_samples={args.n_samples})...")
    saved = 0
    for npz_path in npz_paths:
        if saved >= args.n_samples:
            break
        data = np.load(npz_path, allow_pickle=True)
        N = len(data["agents"])
        for i in range(N):
            if saved >= args.n_samples:
                break
            agent_np    = data["agents"][i]
            gt_traj_np  = data["gt_trajs"][i]
            gt_valid_np = data["gt_valids"][i]
            if not gt_valid_np.any():
                continue
            cond_np = (data["cond_labels"][i] if "cond_labels" in data
                       else _compute_cond_label(agent_np, gt_traj_np, gt_valid_np))
            man_idx = int(cond_np[:6].argmax())
            if not _maneuver_filter(man_idx, args.maneuver):
                continue
            sc_id = str(data["sc_ids"][i])
            shard = os.path.basename(npz_path)
            print(f"\n[{saved+1}/{args.n_samples}] {shard} idx={i}  sc_id={sc_id}  maneuver={MANEUVER_NAMES[man_idx]}")
            result = visualize_one(data, i, models_info, device, out_dir)
            if result:
                saved += 1

    print(f"\n총 {saved}개 저장 완료.")


if __name__ == "__main__":
    main()
