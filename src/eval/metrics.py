"""Trajectory prediction evaluation metrics."""

import numpy as np


def compute_minADE_FDE(pred_traj: np.ndarray, gt_traj: np.ndarray,
                       gt_valid: np.ndarray) -> tuple[float, float]:
    """
    Compute minADE and minFDE over K trajectory hypotheses.

    pred_traj : [K, 80, 2]  or  [80, 2]  (K hypotheses)
    gt_traj   : [80, 2]
    gt_valid  : [80]  bool mask
    Returns   : (minADE_m, minFDE_m)
    """
    if pred_traj.ndim == 2:
        pred_traj = pred_traj[None]   # [1, 80, 2]

    valid_idx = np.where(gt_valid)[0]
    if len(valid_idx) == 0:
        return float("nan"), float("nan")

    ades = np.array([
        np.linalg.norm(pred_traj[k][valid_idx] - gt_traj[valid_idx], axis=1).mean()
        for k in range(len(pred_traj))
    ])
    best_k  = int(np.argmin(ades))
    min_ade = float(ades[best_k])
    last    = int(valid_idx[-1])
    min_fde = float(np.linalg.norm(pred_traj[best_k][last] - gt_traj[last]))
    return min_ade, min_fde


def compute_MR(pred_traj: np.ndarray, gt_traj: np.ndarray,
               gt_valid: np.ndarray, threshold: float = 2.0) -> float:
    """
    Miss Rate (MR): minFDE > threshold 이면 miss로 판정.

    pred_traj : [K, 80, 2]  or  [80, 2]
    gt_traj   : [80, 2]
    gt_valid  : [80] bool
    threshold : m  (기본값 2.0 m)
    Returns   : 0.0 (hit) 또는 1.0 (miss)
    """
    if pred_traj.ndim == 2:
        pred_traj = pred_traj[None]

    valid_idx = np.where(gt_valid)[0]
    if len(valid_idx) == 0:
        return float("nan")

    last = int(valid_idx[-1])
    fdes = np.linalg.norm(pred_traj[:, last] - gt_traj[last], axis=1)  # [K]
    return float(fdes.min() > threshold)
