"""
시나리오 하나에 대해 4개 모델 예측 vs GT 시각화.
  CV / LSTM / RCM / MTR

출력: comparison_<scenario_id>.png  (ego-relative 좌표계)

Run:
    cd waymo_traj
    python visualize_comparison.py \
        --ckpt      checkpoints/model_best.pt \
        --lstm_ckpt checkpoints/lstm_best.pt \
        [--mtr_result ../MTR/output/waymo/mtr_single_gpu/demo_test/eval/eval_with_train/result.pkl] \
        [--split train]   # MTR eval_with_train 결과면 train 사용
        [--scenario_idx 0] [--curve] [--device cuda]

    --split  : 사용할 데이터 분할 (val / train, 기본: val)
    --curve  : 커브(턴) 시나리오를 자동 탐색 (max lateral > 12 m)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.tfrecord       import iter_tfrecords
from src.data.features       import extract_features, extract_risk_label
from src.models.motion_model import RiskConditionedModel
from src.models.baselines    import ConstantVelocityBaseline, LSTMBaseline
from src.eval.metrics        import compute_minADE_FDE


_TFRECORD_PATHS = {
    "val": [
        os.path.join(ROOT, "waymo-motion-v1_3_0", "val",
                     f"validation.tfrecord-{i:05d}-of-00150")
        for i in range(4)
    ],
    "train": [
        os.path.join(ROOT, "waymo-motion-v1_3_0", "train",
                     f"training.tfrecord-{i:05d}-of-01000")
        for i in range(5)
    ],
}

_MTR_DEFAULT = os.path.join(
    ROOT, "..", "MTR", "output", "waymo", "mtr_single_gpu",
    "demo_test", "eval", "eval_with_train", "result.pkl"
)

# RCM 비최적 모드 색상 팔레트
MODE_COLORS_RCM = ["#e07b8a", "#e09b6b", "#d4b84a", "#8abde0", "#a08ae0"]
# MTR 비최적 모드 색상 팔레트
MODE_COLORS_MTR = ["#c5a3e0", "#b08acf", "#d4b8f0", "#a08abf", "#e0c8f0"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         type=str,
                   default=os.path.join(ROOT, "checkpoints/model_best.pt"))
    p.add_argument("--lstm_ckpt",    type=str,
                   default=os.path.join(ROOT, "checkpoints/lstm_best.pt"))
    p.add_argument("--mtr_result",   type=str,
                   default=_MTR_DEFAULT,
                   help="MTR result.pkl 경로 (없으면 MTR 제외)")
    p.add_argument("--split",         type=str, default="val",
                   choices=["val", "train"],
                   help="val tfrecords 또는 train tfrecords 사용 (MTR eval_with_train → train)")
    p.add_argument("--scenario_idx", type=int, default=0,
                   help="몇 번째 유효 시나리오를 시각화할지 (0-based, --curve 없을 때)")
    p.add_argument("--curve",        action="store_true",
                   help="커브 시나리오 자동 탐색 (max |lateral| > 12 m)")
    p.add_argument("--curve_thresh", type=float, default=12.0,
                   help="커브 판단 횡방향 임계값 (m)")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",      type=str, default=".")
    return p.parse_args()


def load_mtr_results(path):
    """result.pkl → {scenario_id: [item, ...]} 딕셔너리."""
    import pickle
    if not os.path.exists(path):
        print(f"[MTR] result.pkl 없음: {path}")
        return None
    with open(path, "rb") as f:
        raw = pickle.load(f)
    by_sc = {}
    for sublist in raw:
        for item in sublist:
            sid = item["scenario_id"]
            by_sc.setdefault(sid, []).append(item)
    print(f"[MTR] {len(by_sc)}개 시나리오 로드 완료: {path}")
    return by_sc


def mtr_pred_to_ego(pred_trajs, x0, y0, cos_h, sin_h):
    """MTR global 예측 → ego-relative 좌표 변환.
    pred_trajs: (K, T, 2) float32  global coords
    반환: (K, T, 2) ego-relative
    """
    dx = pred_trajs[..., 0] - x0
    dy = pred_trajs[..., 1] - y0
    ex =  dx * cos_h + dy * sin_h
    ey = -dx * sin_h + dy * cos_h
    return np.stack([ex, ey], axis=-1)


def get_mtr_pred(sc, mtr_by_id):
    """시나리오 protobuf + MTR 결과 딕셔너리 → ego-relative MTR 예측 (K,80,2) 또는 None."""
    if mtr_by_id is None:
        return None
    items = mtr_by_id.get(sc.scenario_id)
    if not items:
        return None

    sdc_idx = sc.sdc_track_index
    t0      = sc.current_time_index
    ego_s   = sc.tracks[sdc_idx].states[t0]
    x0, y0  = ego_s.center_x, ego_s.center_y
    cos_h   = float(np.cos(ego_s.heading))
    sin_h   = float(np.sin(ego_s.heading))

    # SDC에 해당하는 MTR item 탐색
    mtr_item = None
    for item in items:
        if int(item["track_index_to_predict"]) == sdc_idx:
            mtr_item = item
            break
    if mtr_item is None:
        return None

    pred = mtr_pred_to_ego(mtr_item["pred_trajs"], x0, y0, cos_h, sin_h)
    return pred   # (K, 80, 2)


def _prep_inputs(feats, device):
    agent = feats["agent_tensor"]
    scene = feats["scene_tensor"]
    traf  = feats["traffic_tensor"]
    ego_hist  = agent[0:1]
    social    = agent[1:][None]
    map_scene = scene[None]
    traf_t    = traf[None]
    return (
        torch.from_numpy(ego_hist).to(device),
        torch.from_numpy(social).to(device),
        torch.from_numpy(map_scene).to(device),
        torch.from_numpy(traf_t).to(device),
    )


def draw_map(ax, scene_tensor, alpha=0.4):
    for poly in scene_tensor:
        xs, ys = poly[:, 0], poly[:, 1]
        if np.all(xs == 0) and np.all(ys == 0):
            continue
        ax.plot(xs, ys, color="lightgray", linewidth=0.9, alpha=alpha, zorder=1)


def best_mode_idx(pred_traj, gt_traj, gt_valid):
    valid_idx = np.where(gt_valid)[0]
    if len(valid_idx) == 0:
        return 0
    if pred_traj.ndim == 2:
        return 0
    ades = np.array([
        np.linalg.norm(pred_traj[k][valid_idx] - gt_traj[valid_idx], axis=1).mean()
        for k in range(pred_traj.shape[0])
    ])
    return int(ades.argmin())


def add_time_ticks(ax, traj, valid_idx, color, step=10, zorder=6):
    """10 스텝(=1초)마다 원형 마커 표시."""
    ticks = [i for i in valid_idx if (i + 1) % step == 0]
    if len(ticks):
        ax.scatter(traj[ticks, 0], traj[ticks, 1],
                   color=color, s=35, zorder=zorder, edgecolors="white",
                   linewidths=0.8)


def add_arrow(ax, traj, valid_idx, color, zorder=7):
    """궤적 끝 방향으로 화살표."""
    if len(valid_idx) < 2:
        return
    i0, i1 = valid_idx[-2], valid_idx[-1]
    ax.annotate("", xy=(traj[i1, 0], traj[i1, 1]),
                xytext=(traj[i0, 0], traj[i0, 1]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=zorder)


def compute_view_bounds(arrays, margin=15):
    """여러 궤적에서 x/y 범위 계산."""
    all_pts = np.concatenate([a for a in arrays if a is not None], axis=0)
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # ── MTR 결과 로드 ─────────────────────────────────────────────────────────
    mtr_by_id = load_mtr_results(os.path.abspath(args.mtr_result))
    data_paths = _TFRECORD_PATHS[args.split]
    print(f"[data] split={args.split}, {len(data_paths)} tfrecord(s)")

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    mm_model = RiskConditionedModel(d_model=128, K=6, n_layers=2).to(device)
    mm_label_short = "RCM (random)"
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        mm_model.load_state_dict(ckpt["model"])
        ep  = ckpt.get("epoch", "?")
        ade = ckpt.get("ev_ade", float("nan"))
        mm_label_short = f"RCM ep{ep}"
        print(f"[RCM]  epoch={ep}  val_minADE={ade:.3f}m")
    mm_model.eval()

    lstm_model = LSTMBaseline(input_size=6, hidden_size=64, num_layers=2).to(device)
    lstm_label_short = "LSTM (random)"
    if os.path.exists(args.lstm_ckpt):
        ckpt_l = torch.load(args.lstm_ckpt, map_location=device)
        lstm_model.load_state_dict(ckpt_l["model"])
        ep_l  = ckpt_l.get("epoch", "?")
        ade_l = ckpt_l.get("ev_ade", float("nan"))
        lstm_label_short = f"LSTM ep{ep_l}"
        print(f"[LSTM] epoch={ep_l}  val_ADE={ade_l:.3f}m")
    lstm_model.eval()
    cv_model = ConstantVelocityBaseline()

    # ── 시나리오 탐색 ────────────────────────────────────────────────────────
    from waymo_open_dataset.protos import scenario_pb2

    feats  = None
    sc_id  = ""
    sc_obj = None   # protobuf Scenario (MTR 변환에 필요)
    found  = 0

    mode = "커브 자동탐색" if args.curve else f"idx={args.scenario_idx}"
    print(f"\n시나리오 탐색 중 ({mode})...")

    for raw_bytes in iter_tfrecords(data_paths):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            f             = extract_features(sc)
            risk_label_np = extract_risk_label(sc)
        except Exception:
            continue
        if not f["gt_valid"].any():
            continue

        # MTR가 로드된 경우, SDC가 MTR 예측 대상인 시나리오만 선택
        if mtr_by_id is not None:
            items = mtr_by_id.get(sc.scenario_id, [])
            sdc_in_mtr = any(
                int(it["track_index_to_predict"]) == sc.sdc_track_index
                for it in items
            )
            if not sdc_in_mtr:
                continue

        if args.curve:
            gt = f["gt_trajectory"]
            valid = f["gt_valid"]
            lat_disp = np.abs(gt[valid, 1]).max() if valid.any() else 0
            if lat_disp < args.curve_thresh:
                found += 1
                continue
            feats          = f
            risk_label_sel = risk_label_np
            sc_id          = sc.scenario_id
            sc_obj         = sc
            print(f"커브 시나리오 발견 (idx={found})  max_lateral={lat_disp:.1f}m  ID={sc_id}")
            break
        else:
            if found == args.scenario_idx:
                feats          = f
                risk_label_sel = risk_label_np
                sc_id          = sc.scenario_id
                sc_obj         = sc
                print(f"시나리오 ID: {sc_id}")
                break
            found += 1

    if feats is None:
        print("ERROR: 조건에 맞는 시나리오를 찾지 못했습니다.")
        return

    gt_traj  = feats["gt_trajectory"]   # [80,2]
    gt_valid = feats["gt_valid"]         # [80]
    valid_idx = np.where(gt_valid)[0]

    # ── 예측 ─────────────────────────────────────────────────────────────────
    cv_pred = cv_model.predict(feats["agent_tensor"][0])

    ego_xy = torch.from_numpy(
        feats["agent_tensor"][0, :, :6]   # LSTM은 6-feature (x,y,vx,vy,cos_h,sin_h)로 학습됨
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        lstm_pred = lstm_model(ego_xy)[0].cpu().numpy()

    ego_h, soc, map_scene, traf = _prep_inputs(feats, device)
    risk_gt = torch.from_numpy(risk_label_sel).unsqueeze(0).to(device)  # [1, 3]
    with torch.no_grad():
        out = mm_model(ego_h, soc, map_scene, traf, risk_label=risk_gt, gt_traj=None)
    mm_pred = out["trajectory"][0].cpu().numpy()   # [K,80,2]
    bk_rcm  = best_mode_idx(mm_pred, gt_traj, gt_valid)

    # MTR 예측 (없으면 None)
    mtr_pred = get_mtr_pred(sc_obj, mtr_by_id)   # (K,80,2) or None
    bk_mtr   = best_mode_idx(mtr_pred, gt_traj, gt_valid) if mtr_pred is not None else None

    cv_ade,   cv_fde   = compute_minADE_FDE(cv_pred,  gt_traj, gt_valid)
    lstm_ade, lstm_fde = compute_minADE_FDE(lstm_pred, gt_traj, gt_valid)
    mm_ade,   mm_fde   = compute_minADE_FDE(mm_pred,   gt_traj, gt_valid)
    if mtr_pred is not None:
        mtr_ade, mtr_fde = compute_minADE_FDE(mtr_pred, gt_traj, gt_valid)
    else:
        mtr_ade = mtr_fde = float("nan")

    # ── 캔버스: 왼쪽 전경 / 오른쪽 줌 ──────────────────────────────────────
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 8))
    mtr_tag = f"  MTR minADE={mtr_ade:.2f}m" if mtr_pred is not None else "  MTR: N/A"
    fig.suptitle(
        f"Trajectory Prediction Comparison  |  Scenario: {sc_id}\n"
        f"ego-relative coords  (T=11 hist → 80 future @ 10 Hz = 8 s){mtr_tag}",
        fontsize=12, y=1.01
    )

    ego_hist_xy = feats["agent_tensor"][0, :, :2]   # [11,2]

    # 줌 범위: 모든 궤적 포함 + 여유
    all_trajs = [
        gt_traj[valid_idx],
        cv_pred[valid_idx],
        lstm_pred[valid_idx],
        mm_pred[bk_rcm][valid_idx],
        ego_hist_xy,
    ]
    if mtr_pred is not None:
        all_trajs.append(mtr_pred[bk_mtr][valid_idx])
    zx0, zx1, zy0, zy1 = compute_view_bounds(all_trajs, margin=10)

    for ax, is_zoom in [(ax_full, False), (ax_zoom, True)]:
        # 맵
        draw_map(ax, feats["scene_tensor"], alpha=0.45 if is_zoom else 0.3)

        # 주변 에이전트
        agent = feats["agent_tensor"]
        for i in range(1, agent.shape[0]):
            cx, cy = agent[i, -1, 0], agent[i, -1, 1]
            if cx == 0 and cy == 0:
                continue
            ax.scatter(cx, cy, color="gray", s=20 if is_zoom else 15,
                       alpha=0.5, zorder=3)

        # 에고 히스토리
        ax.plot(ego_hist_xy[:, 0], ego_hist_xy[:, 1],
                color="black", linewidth=2.5, solid_capstyle="round",
                label="Ego history", zorder=7)
        ax.scatter(ego_hist_xy[-1, 0], ego_hist_xy[-1, 1],
                   color="black", s=90, zorder=8)

        # GT future
        ax.plot(gt_traj[valid_idx, 0], gt_traj[valid_idx, 1],
                color="#2ca02c", linewidth=2.5, linestyle="--",
                label="GT future", zorder=8)
        add_time_ticks(ax, gt_traj, valid_idx, "#2ca02c", zorder=9)
        add_arrow(ax, gt_traj, valid_idx, "#2ca02c", zorder=9)

        # Constant Velocity
        ax.plot(cv_pred[valid_idx, 0], cv_pred[valid_idx, 1],
                color="#ff7f0e", linewidth=2, linestyle="-",
                label=f"CV  ADE={cv_ade:.2f}m  FDE={cv_fde:.2f}m", zorder=5)
        add_time_ticks(ax, cv_pred, valid_idx, "#ff7f0e", zorder=6)
        add_arrow(ax, cv_pred, valid_idx, "#ff7f0e", zorder=6)

        # LSTM
        ax.plot(lstm_pred[valid_idx, 0], lstm_pred[valid_idx, 1],
                color="#1f77b4", linewidth=2, linestyle="-",
                label=f"LSTM  ADE={lstm_ade:.2f}m  FDE={lstm_fde:.2f}m", zorder=5)
        add_time_ticks(ax, lstm_pred, valid_idx, "#1f77b4", zorder=6)
        add_arrow(ax, lstm_pred, valid_idx, "#1f77b4", zorder=6)

        # RCM 비최적 모드
        K_rcm = mm_pred.shape[0]
        for k in range(K_rcm):
            if k == bk_rcm:
                continue
            col = MODE_COLORS_RCM[(k % len(MODE_COLORS_RCM))]
            ax.plot(mm_pred[k][valid_idx, 0], mm_pred[k][valid_idx, 1],
                    color=col, linewidth=1.2, linestyle=":", alpha=0.6, zorder=4)

        # RCM 최적 모드
        ax.plot(mm_pred[bk_rcm][valid_idx, 0], mm_pred[bk_rcm][valid_idx, 1],
                color="#d62728", linewidth=2.5, linestyle="-",
                label=f"RCM best  minADE={mm_ade:.2f}m  minFDE={mm_fde:.2f}m",
                zorder=6)
        add_time_ticks(ax, mm_pred[bk_rcm], valid_idx, "#d62728", zorder=7)
        add_arrow(ax, mm_pred[bk_rcm], valid_idx, "#d62728", zorder=7)

        # MTR 비최적 모드
        if mtr_pred is not None:
            K_mtr = mtr_pred.shape[0]
            for k in range(K_mtr):
                if k == bk_mtr:
                    continue
                col = MODE_COLORS_MTR[(k % len(MODE_COLORS_MTR))]
                ax.plot(mtr_pred[k][valid_idx, 0], mtr_pred[k][valid_idx, 1],
                        color=col, linewidth=1.2, linestyle=":", alpha=0.6, zorder=4)

            # MTR 최적 모드
            ax.plot(mtr_pred[bk_mtr][valid_idx, 0], mtr_pred[bk_mtr][valid_idx, 1],
                    color="#9467bd", linewidth=2.5, linestyle="-",
                    label=f"MTR best  minADE={mtr_ade:.2f}m  minFDE={mtr_fde:.2f}m",
                    zorder=6)
            add_time_ticks(ax, mtr_pred[bk_mtr], valid_idx, "#9467bd", zorder=7)
            add_arrow(ax, mtr_pred[bk_mtr], valid_idx, "#9467bd", zorder=7)

        # 축 설정
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)  →  ego forward", fontsize=10)
        ax.set_ylabel("y (m)  →  ego left", fontsize=10)

        if is_zoom:
            ax.set_xlim(zx0, zx1)
            ax.set_ylim(zy0, zy1)
            ax.set_title("Zoom: trajectory area", fontsize=11)
            ax.text(0.02, 0.02,
                    "● = 1 s interval (10 steps)",
                    transform=ax.transAxes, fontsize=8, color="gray",
                    va="bottom")
            extra_patches = []
            extra_patches.append(
                mpatches.Patch(color="gray", alpha=0.5,
                               label=f"RCM other {K_rcm-1} modes (dotted)")
            )
            if mtr_pred is not None:
                extra_patches.append(
                    mpatches.Patch(color="#c5a3e0", alpha=0.6,
                                   label=f"MTR other {K_mtr-1} modes (dotted)")
                )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + extra_patches,
                      labels  + [p.get_label() for p in extra_patches],
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
    out_path = os.path.join(args.out_dir, f"comparison_{sc_id}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n저장: {out_path}")
    print(f"  CV   ADE={cv_ade:.3f}m  FDE={cv_fde:.3f}m")
    print(f"  LSTM ADE={lstm_ade:.3f}m  FDE={lstm_fde:.3f}m")
    print(f"  RCM  minADE={mm_ade:.3f}m  minFDE={mm_fde:.3f}m  (best mode={bk_rcm})")
    if mtr_pred is not None:
        print(f"  MTR  minADE={mtr_ade:.3f}m  minFDE={mtr_fde:.3f}m  (best mode={bk_mtr})")
    else:
        print("  MTR  N/A (시나리오가 MTR 결과에 없음)")


if __name__ == "__main__":
    main()
