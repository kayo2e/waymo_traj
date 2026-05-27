"""
noar_nocond / noar_risk_nolane / noar_maneuver_nolane
세 모델의 실제 궤적 예측을 시나리오 e73b04f426df896d 위에 시각화.

style: visualize_3model_compare.py와 동일 (full map + zoom)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.models.motion_model import RiskConditionedModel
from src.eval.metrics        import compute_minADE_FDE

# ─── 색상 ────────────────────────────────────────────────────────────────────
# noar_nocond  : 회색 계열
COLOR_NOCOND_BEST   = "#636363"
COLORS_NOCOND_OTHER = ["#969696", "#bdbdbd", "#d9d9d9", "#f0f0f0", "#ffffff"]

# noar_risk_nolane : 파란 계열
COLOR_RISK_BEST   = "#2166ac"
COLORS_RISK_OTHER = ["#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"]

# noar_maneuver_nolane : 빨간/주황 계열
COLOR_MAN_BEST   = "#cb181d"
COLORS_MAN_OTHER = ["#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2", "#fff5f0"]


# ─── 모델 로드 ────────────────────────────────────────────────────────────────
def load_model(ckpt_path, cond_dim, device):
    model = RiskConditionedModel(
        agent_dim=10, map_dim=8,
        d_model=128, K=6, n_layers=2,
        use_lane_mamba=False,
        use_risk_prefix=True,
        use_traj_fix=True,
        use_map_per_step=False,
        use_ar=False,
        cond_dim=cond_dim,
    ).to(device)
    label = os.path.basename(os.path.dirname(ckpt_path))
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep  = ckpt.get("epoch", "?")
        ade = ckpt.get("ev_ade", float("nan"))
        label = f"{label} ep{ep}  ADE={ade:.3f}m"
        print(f"  loaded: {label}")
    else:
        print(f"  [WARN] not found: {ckpt_path}")
    model.eval()
    return model, label


def _compute_cond_label(agent, gt_traj, gt_valid):
    label = np.zeros(9, dtype=np.float32)
    vx, vy = agent[0, -1, 2], agent[0, -1, 3]
    v = np.hypot(vx, vy)
    if v < 0.5:
        label[6] = 1.0
    elif v < 5.0:
        label[7] = 1.0
    else:
        label[8] = 1.0
    vi = np.where(gt_valid)[0]
    if len(vi) == 0:
        label[0] = 1.0
        return label
    final_x, final_y = gt_traj[vi[-1]]
    total_dist = np.hypot(final_x, final_y)
    if total_dist < 2.0:
        label[0] = 1.0
    elif abs(final_y) < 3.0:
        label[1] = 1.0
    elif 3.0 <= final_y < 10.0:
        label[2] = 1.0
    elif -10.0 < final_y <= -3.0:
        label[3] = 1.0
    elif final_y >= 10.0:
        label[4] = 1.0
    else:
        label[5] = 1.0
    return label


# ─── 그리기 유틸 ──────────────────────────────────────────────────────────────
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


def compute_bounds(arrays, margin=12):
    pts = np.concatenate([a for a in arrays if a is not None and len(a) > 0])
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


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=str, default="shard_00000.npz")
    p.add_argument("--idx",   type=int, default=51)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(ROOT, "checkpoints")

    print("모델 로드 중...")
    m_nocond, l_nocond = load_model(
        os.path.join(ckpt_dir, "noar_nocond_v2", "model_best.pt"),
        cond_dim=1, device=device)
    m_risk, l_risk = load_model(
        os.path.join(ckpt_dir, "noar_risk_nolane_v2", "model_best.pt"),
        cond_dim=3, device=device)
    m_man, l_man = load_model(
        os.path.join(ckpt_dir, "noar_maneuver_nolane_v2", "model_best.pt"),
        cond_dim=9, device=device)

    # ─── 데이터 로드 ────────────────────────────────────────────────────────
    print(f"\n캐시 로드 중 ({args.shard}, idx={args.idx})...")
    npz = np.load(os.path.join(ROOT, "cache", "val", args.shard),
                  allow_pickle=True)
    idx = args.idx
    sc_id = str(npz["sc_ids"][idx])
    print(f"  sc_id: {sc_id}")

    agent_np   = npz["agents"][idx]    # [32, 11, 10]
    scene_np   = npz["scenes"][idx]    # [50, 10, 6]
    traf_np    = npz["trafs"][idx]     # [6, 1]
    gt_traj_np = npz["gt_trajs"][idx]  # [80, 2]
    gt_valid_np= npz["gt_valids"][idx] # [80]
    risk_np    = npz["risk_labels"][idx]  # [3]

    cond_label_np = _compute_cond_label(agent_np, gt_traj_np, gt_valid_np)  # [9]

    # 텐서 변환
    def t(x): return torch.from_numpy(x).to(device)
    ego_h  = t(agent_np[0:1]).unsqueeze(0)          # [1, 1, 11, 10] → need [1, 11, 10]
    # 실제 모델 입력 shape 확인
    ego_h  = t(agent_np[0]).unsqueeze(0)             # [1, 11, 10]
    soc    = t(agent_np[1:]).unsqueeze(0)            # [1, 31, 11, 10]
    map_s  = t(scene_np).unsqueeze(0)               # [1, 50, 10, 8]
    traf   = t(traf_np).unsqueeze(0)                # [1, 6, 1]

    risk_gt  = t(risk_np).unsqueeze(0)              # [1, 3]
    cond_gt  = t(cond_label_np).unsqueeze(0)        # [1, 9]

    vi = np.where(gt_valid_np)[0]

    print("\n예측 중...")
    with torch.no_grad():
        pred_nocond = m_nocond(ego_h, soc, map_s, traf,
                               risk_label=None)["trajectory"][0].cpu().numpy()      # [6,80,2]
        pred_risk   = m_risk(ego_h, soc, map_s, traf,
                             risk_label=risk_gt)["trajectory"][0].cpu().numpy()     # [6,80,2]
        pred_man    = m_man(ego_h, soc, map_s, traf,
                            risk_label=cond_gt)["trajectory"][0].cpu().numpy()      # [6,80,2]

    ade_nocond, fde_nocond = compute_minADE_FDE(pred_nocond, gt_traj_np, gt_valid_np)
    ade_risk,   fde_risk   = compute_minADE_FDE(pred_risk,   gt_traj_np, gt_valid_np)
    ade_man,    fde_man    = compute_minADE_FDE(pred_man,    gt_traj_np, gt_valid_np)

    ego_hist_xy = agent_np[0, :, :2]  # [11, 2]

    bk_nocond = best_k(pred_nocond, gt_traj_np, vi)
    bk_risk   = best_k(pred_risk,   gt_traj_np, vi)
    bk_man    = best_k(pred_man,    gt_traj_np, vi)

    zx0, zx1, zy0, zy1 = compute_bounds([
        gt_traj_np[vi], ego_hist_xy,
        pred_nocond[bk_nocond][vi],
        pred_risk[bk_risk][vi],
        pred_man[bk_man][vi],
    ], margin=12)

    risk_tags = ["Proximity", "HardBrake", "LaneChange"]
    risk_str  = " | ".join(t for t, v in zip(risk_tags, risk_np) if v) or "No Risk"

    maneuver_names = ["Stop","GoStraight","LCLeft","LCRight","TurnLeft","TurnRight"]
    speed_names    = ["Stopped","Slow","Fast"]
    man_idx = int(cond_label_np[:6].argmax())
    spd_idx = int(cond_label_np[6:].argmax())
    man_str = f"{maneuver_names[man_idx]}/{speed_names[spd_idx]}"

    # ─── 그림 ───────────────────────────────────────────────────────────────
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"Ablation Comparison  |  Scenario: {sc_id}  |  Risk: {risk_str}  |  Maneuver: {man_str}\n"
        f"ego-relative coords  (T=11 hist → 80 future @ 10 Hz = 8 s)",
        fontsize=12, y=1.01
    )

    for ax, is_zoom in [(ax_full, False), (ax_zoom, True)]:
        draw_map(ax, scene_np, alpha=0.45 if is_zoom else 0.3)

        # 주변 에이전트
        for i in range(1, agent_np.shape[0]):
            cx, cy = agent_np[i, -1, 0], agent_np[i, -1, 1]
            if cx == 0 and cy == 0:
                continue
            ax.scatter(cx, cy, color="gray",
                       s=20 if is_zoom else 12, alpha=0.4, zorder=3)

        # ego history
        ax.plot(ego_hist_xy[:, 0], ego_hist_xy[:, 1],
                color="black", linewidth=2.5, solid_capstyle="round",
                label="Ego history", zorder=9)
        ax.scatter(ego_hist_xy[-1, 0], ego_hist_xy[-1, 1],
                   color="black", s=90, zorder=10)

        # GT
        ax.plot(gt_traj_np[vi, 0], gt_traj_np[vi, 1],
                color="#2ca02c", linewidth=2.5, linestyle="--",
                label="GT future", zorder=8)
        add_time_ticks(ax, gt_traj_np, vi, "#2ca02c", zorder=9)
        add_arrow(ax, gt_traj_np, vi, "#2ca02c", zorder=9)

        # 3 모델
        draw_model(ax, pred_nocond, gt_traj_np, vi,
                   COLOR_NOCOND_BEST, COLORS_NOCOND_OTHER,
                   f"NoCond  ADE={ade_nocond:.2f}m  FDE={fde_nocond:.2f}m",
                   zorder_base=4)
        draw_model(ax, pred_risk, gt_traj_np, vi,
                   COLOR_RISK_BEST, COLORS_RISK_OTHER,
                   f"Risk  ADE={ade_risk:.2f}m  FDE={fde_risk:.2f}m",
                   zorder_base=5)
        draw_model(ax, pred_man, gt_traj_np, vi,
                   COLOR_MAN_BEST, COLORS_MAN_OTHER,
                   f"Maneuver★  ADE={ade_man:.2f}m  FDE={fde_man:.2f}m",
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
                               label=f"NoCond other {pred_nocond.shape[0]-1} modes"),
                mpatches.Patch(color="#6baed6", alpha=0.7,
                               label=f"Risk other {pred_risk.shape[0]-1} modes"),
                mpatches.Patch(color="#fb6a4a", alpha=0.7,
                               label=f"Maneuver other {pred_man.shape[0]-1} modes"),
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
    out_path = os.path.join(ROOT, f"3model_compare_{sc_id}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print(f"  NoCond      minADE={ade_nocond:.3f}m  minFDE={fde_nocond:.3f}m")
    print(f"  Risk        minADE={ade_risk:.3f}m  minFDE={fde_risk:.3f}m")
    print(f"  Maneuver    minADE={ade_man:.3f}m  minFDE={fde_man:.3f}m")


if __name__ == "__main__":
    main()
