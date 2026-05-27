"""
Risk prefix conditioning 효과 시각화.

각 리스크 유형(Proximity / HardBrake / LaneChange)별 시나리오에서
  - risk_label 그대로 → 실선
  - risk_label = [0,0,0] (zeroed) → 점선
두 예측을 겹쳐 그려 prefix token의 실제 conditioning 효과를 확인.

Run:
    python visualize_risk_ablation.py
    python visualize_risk_ablation.py --ckpt checkpoints/traj_fix/model_best.pt
    python visualize_risk_ablation.py --n_per_risk 2
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
from src.eval.metrics        import compute_minADE_FDE

_VAL_PATHS = [
    os.path.join(ROOT, "waymo-motion-v1_3_0", "val",
                 f"validation.tfrecord-{i:05d}-of-00150")
    for i in range(8)
]

RISK_NAMES  = ["Proximity", "HardBrake", "LaneChange"]
RISK_COLORS = ["#d62728", "#ff7f0e", "#9467bd"]   # 리스크 유형별 색상


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(ROOT, "checkpoints/traj_fix/model_best.pt"))
    p.add_argument("--n_per_risk", type=int, default=1,
                   help="리스크 유형당 수집할 시나리오 수")
    p.add_argument("--max_scan",   type=int, default=3000,
                   help="탐색할 최대 시나리오 수")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",    type=str, default=".")
    return p.parse_args()


def load_model(ckpt_path, device):
    model = RiskConditionedModel(
        d_model=128, K=6, n_layers=2,
        use_lane_mamba=False,
        use_risk_prefix=True,
        use_traj_fix=True,
        use_map_per_step=False,
    ).to(device)
    label = "RCM (random init)"
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep    = ckpt.get("epoch", "?")
        ade   = ckpt.get("ev_ade", float("nan"))
        label = f"RCM traj_fix  ep{ep}  val ADE={ade:.3f}m"
        print(f"Loaded: {label}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}")
    model.eval()
    return model, label


def _prep(feats, device):
    agent = feats["agent_tensor"]
    scene = feats["scene_tensor"]
    traf  = feats["traffic_tensor"]
    return (
        torch.from_numpy(agent[0:1]).to(device),
        torch.from_numpy(agent[1:][None]).to(device),
        torch.from_numpy(scene[None]).to(device),
        torch.from_numpy(traf[None]).to(device),
    )


def draw_map(ax, scene_tensor, alpha=0.35):
    for poly in scene_tensor:
        xs, ys = poly[:, 0], poly[:, 1]
        if np.all(xs == 0) and np.all(ys == 0):
            continue
        ax.plot(xs, ys, color="lightgray", linewidth=0.8, alpha=alpha, zorder=1)


def add_arrow(ax, traj, vi, color, zorder=7):
    if len(vi) < 2:
        return
    i0, i1 = vi[-2], vi[-1]
    ax.annotate("", xy=(traj[i1, 0], traj[i1, 1]),
                xytext=(traj[i0, 0], traj[i0, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, mutation_scale=13),
                zorder=zorder)


def best_k(pred, gt, vi):
    return int(np.argmin([
        np.linalg.norm(pred[k][vi] - gt[vi], axis=1).mean()
        for k in range(pred.shape[0])
    ]))


def draw_panel(ax, feats, pred_risk, pred_zero, gt_traj, gt_valid,
               risk_name, risk_color, sc_id, is_zoom, xlim=None, ylim=None):
    vi = np.where(gt_valid)[0]
    ego_xy = feats["agent_tensor"][0, :, :2]

    draw_map(ax, feats["scene_tensor"], alpha=0.45 if is_zoom else 0.3)

    # 주변 에이전트
    agent = feats["agent_tensor"]
    for i in range(1, agent.shape[0]):
        cx, cy = agent[i, -1, 0], agent[i, -1, 1]
        if cx == 0 and cy == 0:
            continue
        ax.scatter(cx, cy, color="gray", s=14, alpha=0.35, zorder=2)

    # 에고 히스토리
    ax.plot(ego_xy[:, 0], ego_xy[:, 1],
            color="black", linewidth=2.2, solid_capstyle="round",
            label="Ego history", zorder=9)
    ax.scatter(ego_xy[-1, 0], ego_xy[-1, 1], color="black", s=80, zorder=10)

    # GT
    ax.plot(gt_traj[vi, 0], gt_traj[vi, 1],
            color="#2ca02c", linewidth=2.2, linestyle="--",
            label="GT future", zorder=8)
    add_arrow(ax, gt_traj, vi, "#2ca02c", zorder=9)

    # with risk (실선) — best mode + others
    bk_r = best_k(pred_risk, gt_traj, vi)
    bk_z = best_k(pred_zero, gt_traj, vi)

    for k in range(pred_risk.shape[0]):
        alpha = 0.9 if k == bk_r else 0.25
        lw    = 2.2 if k == bk_r else 0.8
        ls    = "-"
        label = f"risk={risk_name} (ADE={compute_minADE_FDE(pred_risk, gt_traj, gt_valid)[0]:.2f}m)" if k == bk_r else None
        ax.plot(pred_risk[k][vi, 0], pred_risk[k][vi, 1],
                color=risk_color, linewidth=lw, linestyle=ls,
                alpha=alpha, label=label, zorder=5 + (1 if k == bk_r else 0))
    add_arrow(ax, pred_risk[bk_r], vi, risk_color, zorder=7)

    # zeroed risk (점선) — best mode + others
    for k in range(pred_zero.shape[0]):
        alpha = 0.9 if k == bk_z else 0.25
        lw    = 2.2 if k == bk_z else 0.8
        label = f"risk=zeros  (ADE={compute_minADE_FDE(pred_zero, gt_traj, gt_valid)[0]:.2f}m)" if k == bk_z else None
        ax.plot(pred_zero[k][vi, 0], pred_zero[k][vi, 1],
                color="#1f77b4", linewidth=lw, linestyle=":",
                alpha=alpha, label=label, zorder=4 + (1 if k == bk_z else 0))
    add_arrow(ax, pred_zero[bk_z], vi, "#1f77b4", zorder=6)

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)

    if is_zoom and xlim:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f"[{risk_name}]  {sc_id[:16]}  — zoom", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax.set_title(f"[{risk_name}]  {sc_id[:16]}  — full map", fontsize=10)


def main():
    args   = parse_args()
    device = torch.device(args.device)
    model, model_label = load_model(args.ckpt, device)

    from waymo_open_dataset.protos import scenario_pb2

    # 리스크 유형별로 시나리오 수집
    # buckets[i] = list of (feats, risk_np, sc_id)  for risk type i
    buckets = [[] for _ in range(3)]
    n_scan  = 0

    print(f"\n시나리오 탐색 중 (최대 {args.max_scan}개)...")
    for raw_bytes in iter_tfrecords(_VAL_PATHS):
        if n_scan >= args.max_scan:
            break
        if all(len(b) >= args.n_per_risk for b in buckets):
            break

        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        n_scan += 1

        try:
            f  = extract_features(sc)
            rl = extract_risk_label(sc)
        except Exception:
            continue
        if not f["gt_valid"].any():
            continue

        for i in range(3):
            if rl[i] and len(buckets[i]) < args.n_per_risk:
                # 단일 리스크 시나리오 우선 (다른 리스크 없는 것)
                if rl.sum() == 1:
                    buckets[i].append((f, rl, sc.scenario_id))
                    print(f"  [{RISK_NAMES[i]}] {sc.scenario_id}  risk={rl}")
                    break

    # 단일 리스크만으로 못 채웠으면 복합 리스크도 허용
    n_scan2 = 0
    if any(len(b) < args.n_per_risk for b in buckets):
        print("  단일 리스크 부족 → 복합 리스크 시나리오 추가 탐색...")
        for raw_bytes in iter_tfrecords(_VAL_PATHS):
            if n_scan2 >= args.max_scan:
                break
            if all(len(b) >= args.n_per_risk for b in buckets):
                break
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(raw_bytes)
            n_scan2 += 1
            try:
                f  = extract_features(sc)
                rl = extract_risk_label(sc)
            except Exception:
                continue
            if not f["gt_valid"].any():
                continue
            for i in range(3):
                if rl[i] and len(buckets[i]) < args.n_per_risk:
                    buckets[i].append((f, rl, sc.scenario_id))
                    print(f"  [{RISK_NAMES[i]}] {sc.scenario_id}  risk={rl}")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    n_found = sum(len(b) for b in buckets)
    if n_found == 0:
        print("ERROR: 리스크 시나리오를 찾지 못했습니다.")
        return

    # 리스크 유형별로 figure 1개씩 (full + zoom)
    for ri, (risk_name, risk_color) in enumerate(zip(RISK_NAMES, RISK_COLORS)):
        if not buckets[ri]:
            print(f"[WARN] {risk_name} 시나리오 없음")
            continue

        for fi, (feats, risk_np, sc_id) in enumerate(buckets[ri]):
            ego_h, soc, map_s, traf = _prep(feats, device)
            risk_gt   = torch.from_numpy(risk_np).unsqueeze(0).float().to(device)
            risk_zero = torch.zeros_like(risk_gt)

            with torch.no_grad():
                pred_risk = model(ego_h, soc, map_s, traf,
                                  risk_label=risk_gt  )["trajectory"][0].cpu().numpy()
                pred_zero = model(ego_h, soc, map_s, traf,
                                  risk_label=risk_zero)["trajectory"][0].cpu().numpy()

            gt_traj  = feats["gt_trajectory"]
            gt_valid = feats["gt_valid"]
            vi       = np.where(gt_valid)[0]

            ade_r, fde_r = compute_minADE_FDE(pred_risk, gt_traj, gt_valid)
            ade_z, fde_z = compute_minADE_FDE(pred_zero, gt_traj, gt_valid)

            ego_xy = feats["agent_tensor"][0, :, :2]
            bk_r   = best_k(pred_risk, gt_traj, vi)
            bk_z   = best_k(pred_zero, gt_traj, vi)

            # zoom 범위
            all_pts = np.concatenate([
                gt_traj[vi], ego_xy,
                pred_risk[bk_r][vi], pred_zero[bk_z][vi],
            ])
            margin = 12
            xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
            ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

            risk_tag = " + ".join(RISK_NAMES[j] for j in range(3) if risk_np[j])

            fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle(
                f"Risk Prefix Ablation  |  {risk_tag}  |  Scenario: {sc_id}\n"
                f"{model_label}\n"
                f"실선=risk label 사용  점선=zeros 입력  (best mode 강조)",
                fontsize=11, y=1.02
            )

            draw_panel(ax_full, feats, pred_risk, pred_zero,
                       gt_traj, gt_valid, risk_name, risk_color,
                       sc_id, is_zoom=False)

            draw_panel(ax_zoom, feats, pred_risk, pred_zero,
                       gt_traj, gt_valid, risk_name, risk_color,
                       sc_id, is_zoom=True, xlim=xlim, ylim=ylim)

            # full map에 zoom box
            from matplotlib.patches import Rectangle
            rect = Rectangle((xlim[0], ylim[0]),
                              xlim[1]-xlim[0], ylim[1]-ylim[0],
                              linewidth=1.5, edgecolor="dimgray",
                              facecolor="none", linestyle="--", zorder=10)
            ax_full.add_patch(rect)

            plt.tight_layout()
            fname = f"risk_ablation_{risk_name.lower()}_{sc_id[:16]}.png"
            out_path = os.path.join(args.out_dir, fname)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"\n[{risk_name}] Saved: {out_path}")
            print(f"  with risk  minADE={ade_r:.3f}m  minFDE={fde_r:.3f}m")
            print(f"  zeros      minADE={ade_z:.3f}m  minFDE={fde_z:.3f}m")
            delta = ade_z - ade_r
            print(f"  Δ ADE (risk - zeros) = {delta:+.3f}m  "
                  f"({'risk 도움됨 ✓' if delta > 0 else 'risk 효과 없음 또는 역효과 ✗'})")


if __name__ == "__main__":
    main()
