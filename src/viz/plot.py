"""Trajectory visualisation utilities."""

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(
    gt_traj: np.ndarray,
    ml_traj_k: np.ndarray,
    best_k: int,
    gemini_traj: np.ndarray | None = None,
    map_features=None,
    ego_pose: tuple | None = None,
    ade_ml: float | None = None,
    fde_ml: float | None = None,
    ade_gemini: float | None = None,
    fde_gemini: float | None = None,
    scenario_id: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot ML (best mode) and optionally Gemini trajectories against GT.

    gt_traj    : [80, 2]    ego-relative GT
    ml_traj_k  : [K, 80, 2] K predicted hypotheses
    best_k     : int         index of best mode (by minADE)
    gemini_traj: [80, 2] or None
    map_features: list of WOMD proto map_features or None
    ego_pose   : (x0, y0, theta) world-frame ego pose for map drawing
    """
    traj_best = ml_traj_k[best_k]    # [80, 2]
    n_panels  = 2 if gemini_traj is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    ade_ml_str     = f"  minADE={ade_ml:.2f}m" if ade_ml is not None else ""
    fde_ml_str     = f"  minFDE={fde_ml:.2f}m" if fde_ml is not None else ""
    ade_gem_str    = f"  ADE={ade_gemini:.2f}m" if ade_gemini is not None else ""
    fde_gem_str    = f"  FDE={fde_gemini:.2f}m" if fde_gemini is not None else ""

    fig.suptitle(
        f"Trajectory Prediction — Scenario {scenario_id[:8]}…\n"
        f"ML:{ade_ml_str}{fde_ml_str}  |  Gemini:{ade_gem_str}{fde_gem_str}",
        fontsize=12, fontweight="bold",
    )

    def _draw_base(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (forward, m)")
        ax.set_ylabel("y (left, m)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)

        # Map polylines
        if map_features is not None and ego_pose is not None:
            x0, y0, theta = ego_pose
            cos_h, sin_h  = np.cos(theta), np.sin(theta)
            for feat in map_features[:30]:
                try:
                    if feat.HasField("lane"):
                        pts = [(p.x, p.y) for p in feat.lane.polyline]
                    elif feat.HasField("road_edge"):
                        pts = [(p.x, p.y) for p in feat.road_edge.polyline]
                    elif feat.HasField("road_line"):
                        pts = [(p.x, p.y) for p in feat.road_line.polyline]
                    else:
                        continue
                    if len(pts) < 2:
                        continue
                    wx = np.array([p[0] for p in pts]) - x0
                    wy = np.array([p[1] for p in pts]) - y0
                    ex =  wx * cos_h + wy * sin_h
                    ey = -wx * sin_h + wy * cos_h
                    ax.plot(ex, ey, color="lightgray", lw=1, zorder=1)
                except Exception:
                    pass

        ax.plot(0, 0, marker="^", ms=14, color="black", zorder=10, label="Ego (now)")

        valid = ~np.any(np.isnan(gt_traj), axis=1)
        ax.plot(gt_traj[valid, 0], gt_traj[valid, 1],
                color="green", lw=2.5, zorder=5, label="Ground Truth")
        for label, step in [("1s", 9), ("3s", 29), ("5s", 49), ("8s", 79)]:
            if step < len(gt_traj) and valid[step]:
                ax.scatter(gt_traj[step, 0], gt_traj[step, 1], color="green", s=60, zorder=6)
                ax.annotate(label, gt_traj[step], textcoords="offset points",
                            xytext=(4, 4), fontsize=8, color="green")

    # ── Panel 0: ML best mode ──────────────────────────────────────────────
    _draw_base(axes[0], f"WaymoMotionModel K={len(ml_traj_k)} (best mode {best_k}){ade_ml_str}{fde_ml_str}")
    # Draw all K modes faintly
    for k in range(len(ml_traj_k)):
        alpha = 0.6 if k == best_k else 0.15
        lw    = 2.0 if k == best_k else 0.8
        axes[0].plot(ml_traj_k[k, :, 0], ml_traj_k[k, :, 1],
                     color="tomato", lw=lw, alpha=alpha, zorder=4)
    for label, step in [("1s", 9), ("3s", 29), ("5s", 49), ("8s", 79)]:
        axes[0].scatter(traj_best[step, 0], traj_best[step, 1],
                        color="tomato", s=60, zorder=6)
        axes[0].annotate(label, traj_best[step], textcoords="offset points",
                         xytext=(4, 4), fontsize=8, color="tomato")
    axes[0].legend(loc="best", fontsize=9)

    # ── Panel 1: Gemini ────────────────────────────────────────────────────
    if gemini_traj is not None:
        _draw_base(axes[1], f"Gemini Stage 3-C{ade_gem_str}{fde_gem_str}")
        axes[1].plot(gemini_traj[:, 0], gemini_traj[:, 1],
                     color="royalblue", lw=2, zorder=5, label="Gemini")
        for label, step in [("1s", 9), ("3s", 29), ("5s", 49), ("8s", 79)]:
            axes[1].scatter(gemini_traj[step, 0], gemini_traj[step, 1],
                            color="royalblue", s=60, zorder=6)
            axes[1].annotate(label, gemini_traj[step], textcoords="offset points",
                             xytext=(4, 4), fontsize=8, color="royalblue")
        axes[1].legend(loc="best", fontsize=9)

    # Sync axis limits
    all_x = np.concatenate([gt_traj[~np.isnan(gt_traj[:, 0]), 0], traj_best[:, 0], [0]])
    all_y = np.concatenate([gt_traj[~np.isnan(gt_traj[:, 1]), 1], traj_best[:, 1], [0]])
    if gemini_traj is not None:
        all_x = np.concatenate([all_x, gemini_traj[:, 0]])
        all_y = np.concatenate([all_y, gemini_traj[:, 1]])
    m = 3.0
    for ax in axes:
        ax.set_xlim(all_x.min() - m, all_x.max() + m)
        ax.set_ylim(all_y.min() - m, all_y.max() + m)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    return fig
