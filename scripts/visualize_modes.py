"""
RCM K=6 모드 다양성 시각화.

출력 (두 개 파일):
  mode_diversity_grid.png  — N개 시나리오 그리드 (각 패널에 6 모드 전부)
  mode_diversity_stats.png — 모드 종점 scatter + pairwise 거리 분포 + best-mode 빈도

Run:
    cd waymo_traj
    python visualize_modes.py \
        --ckpt checkpoints/model_best.pt \
        [--n 20] [--device cuda]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.tfrecord       import iter_tfrecords
from src.data.features       import extract_features
from src.models.motion_model import RiskConditionedModel

VAL_PATHS = [
    os.path.join(ROOT, "waymo-motion-v1_3_0", "val",
                 f"validation.tfrecord-{i:05d}-of-00150")
    for i in range(4)
]

MODE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   type=str,
                   default=os.path.join(ROOT, "checkpoints/model_best.pt"))
    p.add_argument("--n",      type=int, default=20,
                   help="시각화할 시나리오 수")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default=".")
    return p.parse_args()


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


def collect_predictions(model, device, n):
    """n개 시나리오에서 RCM 6-mode 예측 수집."""
    from waymo_open_dataset.protos import scenario_pb2

    records = []
    for raw_bytes in iter_tfrecords(VAL_PATHS):
        if len(records) >= n:
            break
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            feats = extract_features(sc)
        except Exception:
            continue
        if not feats["gt_valid"].any():
            continue

        ego_h, soc, map_scene, traf = _prep_inputs(feats, device)
        with torch.no_grad():
            out = model(ego_h, soc, map_scene, traf, risk_label=None, gt_traj=None)
        mm_pred = out["trajectory"][0].cpu().numpy()   # [K, 80, 2]

        gt_traj  = feats["gt_trajectory"]   # [80, 2]
        gt_valid = feats["gt_valid"]         # [80]
        valid_idx = np.where(gt_valid)[0]

        # best mode (oracle)
        ades = np.array([
            np.linalg.norm(mm_pred[k][valid_idx] - gt_traj[valid_idx], axis=1).mean()
            for k in range(mm_pred.shape[0])
        ])
        best_k = int(ades.argmin())

        records.append({
            "pred":      mm_pred,         # [K, 80, 2]
            "gt":        gt_traj,         # [80, 2]
            "valid":     valid_idx,
            "ego_hist":  feats["agent_tensor"][0, :, :2],  # [11, 2]
            "best_k":    best_k,
            "ades":      ades,
        })
        if len(records) % 5 == 0:
            print(f"  수집: {len(records)}/{n}")

    print(f"총 {len(records)}개 수집 완료")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 1. 시나리오 그리드
# ─────────────────────────────────────────────────────────────────────────────
def plot_grid(records, out_path):
    n    = len(records)
    cols = 5
    rows = (n + cols - 1) // cols
    K    = records[0]["pred"].shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for i, rec in enumerate(records):
        ax  = axes[i]
        gt  = rec["gt"]
        vid = rec["valid"]
        hist = rec["ego_hist"]
        pred = rec["pred"]   # [K, 80, 2]

        # ego history
        ax.plot(hist[:, 0], hist[:, 1], color="black", lw=1.5, zorder=5)
        ax.scatter(0, 0, color="black", s=40, zorder=6)

        # GT
        ax.plot(gt[vid, 0], gt[vid, 1], color="lime", lw=2,
                linestyle="--", zorder=7)

        # 6 modes
        for k in range(K):
            col    = MODE_COLORS[k % len(MODE_COLORS)]
            lw     = 2.5 if k == rec["best_k"] else 1.0
            alpha  = 1.0 if k == rec["best_k"] else 0.55
            ls     = "-"  if k == rec["best_k"] else ":"
            ax.plot(pred[k][vid, 0], pred[k][vid, 1],
                    color=col, lw=lw, alpha=alpha, linestyle=ls, zorder=4)
            # 종점 마커
            ax.scatter(pred[k][vid[-1], 0], pred[k][vid[-1], 1],
                       color=col, s=25, zorder=8, edgecolors="white", lw=0.5)

        ax.set_aspect("equal")
        ax.set_title(f"#{i}  best={rec['best_k']}  ADE={rec['ades'].min():.2f}m",
                     fontsize=8)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # 사용되지 않는 패널 숨김
    for j in range(len(records), len(axes)):
        axes[j].set_visible(False)

    # 공통 범례
    handles = [
        plt.Line2D([0], [0], color=MODE_COLORS[k], lw=2,
                   linestyle="-" if k == 0 else ":",
                   label=f"Mode {k}" + (" (best)" if k == 0 else ""))
        for k in range(K)
    ] + [
        plt.Line2D([0], [0], color="lime",  lw=2, linestyle="--", label="GT"),
        plt.Line2D([0], [0], color="black", lw=2, label="Ego hist"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=K + 2, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"RCM K={K} Mode Diversity — {n} Scenarios\n"
                 "(solid thick = oracle best mode, dotted = others)",
                 fontsize=13, y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid 저장 → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 통계 패널
# ─────────────────────────────────────────────────────────────────────────────
def plot_stats(records, out_path):
    K = records[0]["pred"].shape[0]
    n = len(records)

    # ── 데이터 수집 ──────────────────────────────────────────────────────────
    # 종점 (valid 마지막 인덱스) 좌표 [N, K, 2]
    endpoints = np.array([
        [rec["pred"][k][rec["valid"][-1]] for k in range(K)]
        for rec in records
    ])  # [N, K, 2]

    # pairwise 거리 (per scenario): K*(K-1)/2 pairs
    pw_dists = []
    for rec in records:
        vid = rec["valid"]
        for ka in range(K):
            for kb in range(ka + 1, K):
                d = np.linalg.norm(
                    rec["pred"][ka][vid[-1]] - rec["pred"][kb][vid[-1]]
                )
                pw_dists.append(d)
    pw_dists = np.array(pw_dists)

    # best mode 빈도
    best_freq = np.bincount([rec["best_k"] for rec in records], minlength=K)

    # mode별 ADE (oracle 아님, 각 모드 독립적으로)
    mode_ades = np.array([
        [rec["ades"][k] for rec in records] for k in range(K)
    ])  # [K, N]

    # ── 그림 레이아웃 ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    # (A) 종점 scatter — 모드별 색상
    ax_ep = fig.add_subplot(gs[0, :2])
    for k in range(K):
        ax_ep.scatter(endpoints[:, k, 0], endpoints[:, k, 1],
                      color=MODE_COLORS[k], s=30, alpha=0.6,
                      label=f"Mode {k}", zorder=3)
    ax_ep.scatter(0, 0, color="black", s=80, marker="x", zorder=5, label="Ego t=0")
    ax_ep.set_title(f"Mode Endpoints (final step) — {n} scenarios", fontsize=11)
    ax_ep.set_xlabel("x (m)  forward", fontsize=9)
    ax_ep.set_ylabel("y (m)  left", fontsize=9)
    ax_ep.set_aspect("equal")
    ax_ep.grid(True, alpha=0.25)
    ax_ep.legend(fontsize=8, ncol=2)

    # (B) Pairwise 모드 거리 히스토그램
    ax_pw = fig.add_subplot(gs[0, 2])
    ax_pw.hist(pw_dists, bins=40, color="#4daf4a", edgecolor="white", alpha=0.8)
    ax_pw.axvline(pw_dists.mean(), color="red", lw=1.5,
                  label=f"mean={pw_dists.mean():.2f}m")
    ax_pw.axvline(np.median(pw_dists), color="orange", lw=1.5,
                  label=f"med={np.median(pw_dists):.2f}m")
    ax_pw.set_title("Pairwise Mode Distance\n(at final timestep)", fontsize=10)
    ax_pw.set_xlabel("distance (m)", fontsize=9)
    ax_pw.legend(fontsize=8)
    ax_pw.grid(True, alpha=0.25)

    # (C) Best mode 빈도
    ax_bk = fig.add_subplot(gs[0, 3])
    bars = ax_bk.bar(range(K), best_freq, color=MODE_COLORS[:K], edgecolor="white")
    ax_bk.set_xticks(range(K))
    ax_bk.set_xticklabels([f"M{k}" for k in range(K)])
    ax_bk.set_title("Oracle Best Mode Frequency", fontsize=10)
    ax_bk.set_ylabel("count", fontsize=9)
    # 균등 분포 기준선
    ax_bk.axhline(n / K, color="gray", linestyle="--", lw=1.2,
                  label=f"uniform = {n/K:.1f}")
    ax_bk.legend(fontsize=8)
    ax_bk.grid(True, alpha=0.25, axis="y")
    for bar, cnt in zip(bars, best_freq):
        ax_bk.text(bar.get_x() + bar.get_width() / 2, cnt + 0.3,
                   str(cnt), ha="center", va="bottom", fontsize=9)

    # (D) 모드별 ADE boxplot
    ax_ad = fig.add_subplot(gs[1, :2])
    bp = ax_ad.boxplot(mode_ades.T, patch_artist=True,
                       medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp["boxes"], MODE_COLORS[:K]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax_ad.set_xticks(range(1, K + 1))
    ax_ad.set_xticklabels([f"Mode {k}" for k in range(K)])
    ax_ad.set_ylabel("ADE (m)", fontsize=9)
    ax_ad.set_title("Per-Mode ADE Distribution (non-oracle)", fontsize=10)
    ax_ad.grid(True, alpha=0.25, axis="y")

    # (E) 각 시나리오에서 최악 모드 vs 최선 모드 ADE 차이
    spread = mode_ades.max(axis=0) - mode_ades.min(axis=0)   # [N]
    ax_sp = fig.add_subplot(gs[1, 2])
    ax_sp.hist(spread, bins=30, color="#984ea3", edgecolor="white", alpha=0.8)
    ax_sp.axvline(spread.mean(), color="red", lw=1.5,
                  label=f"mean={spread.mean():.2f}m")
    ax_sp.set_title("ADE Spread per Scenario\n(max_mode - min_mode ADE)", fontsize=10)
    ax_sp.set_xlabel("ADE spread (m)", fontsize=9)
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.25)

    # (F) 숫자 요약
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.axis("off")
    minADE_oracle = np.array([rec["ades"].min() for rec in records]).mean()
    mean_ADE_all  = mode_ades.mean()
    freq_pct = best_freq / n * 100

    summary = (
        f"N scenarios : {n}\n"
        f"K modes     : {K}\n\n"
        f"minADE (oracle) : {minADE_oracle:.3f} m\n"
        f"Mean ADE (all)  : {mean_ADE_all:.3f} m\n\n"
        f"Pairwise dist\n"
        f"  mean  : {pw_dists.mean():.2f} m\n"
        f"  median: {np.median(pw_dists):.2f} m\n"
        f"  <2 m  : {(pw_dists < 2).mean()*100:.1f}%\n\n"
        f"Best mode freq (%):\n"
        + "\n".join(f"  Mode {k}: {freq_pct[k]:.1f}%" for k in range(K))
        + f"\n\nADE spread\n  mean: {spread.mean():.2f} m"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax_txt.set_title("Summary", fontsize=10)

    fig.suptitle(f"RCM Mode Diversity Analysis  ({n} val scenarios)",
                 fontsize=14, y=1.01)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Stats 저장 → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device)

    model = RiskConditionedModel(d_model=128, K=6, n_layers=2).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep  = ckpt.get("epoch", "?")
        ade = ckpt.get("ev_ade", float("nan"))
        print(f"[RCM] epoch={ep}  val_minADE={ade:.3f}m")
    else:
        print(f"[RCM] 체크포인트 없음 — random init  ({args.ckpt})")
    model.eval()

    print(f"\n시나리오 {args.n}개 수집 중...")
    records = collect_predictions(model, device, args.n)

    grid_path  = os.path.join(args.out_dir, "mode_diversity_grid.png")
    stats_path = os.path.join(args.out_dir, "mode_diversity_stats.png")

    plot_grid(records, grid_path)
    plot_stats(records, stats_path)

    print("\n완료.")


if __name__ == "__main__":
    main()
