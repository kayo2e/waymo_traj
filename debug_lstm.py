"""
LSTM 학습 진단 스크립트.

확인 항목:
  1. 데이터 sanity check: ego 역사, GT 궤적이 물리적으로 맞는가?
  2. LSTM이 학습 중에 loss가 실제로 줄어드는가?
  3. 학습 후 예측이 GT와 비슷한 방향/크기를 보이는가?
  4. Constant Velocity vs LSTM 비교

Run:
    cd waymo_traj
    python debug_lstm.py [--n_scenarios 200] [--epochs 3] [--device cpu]
"""

import argparse
import os
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "waymo-motion-v1_3_0")
sys.path.insert(0, ROOT)

from src.data.tfrecord    import iter_tfrecords
from src.data.features    import extract_features
from src.models.baselines import LSTMBaseline, ConstantVelocityBaseline

TRAIN_PATHS = [
    os.path.join(DATA_ROOT, "train", f"training.tfrecord-{i:05d}-of-01000")
    for i in range(5)   # 빠른 진단용: 5 shard만
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_scenarios", type=int, default=200)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터 수집
# ──────────────────────────────────────────────────────────────────────────────
def collect_data(n_scenarios):
    from waymo_open_dataset.protos import scenario_pb2

    records = []
    for raw_bytes in iter_tfrecords(TRAIN_PATHS):
        if len(records) >= n_scenarios:
            break
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            feats = extract_features(sc)
        except Exception:
            continue
        if not feats["gt_valid"].any():
            continue
        records.append(feats)

    print(f"\n[데이터] {len(records)}개 시나리오 로드 완료")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 2. Sanity check
# ──────────────────────────────────────────────────────────────────────────────
def sanity_check(records):
    print("\n" + "="*60)
    print("[SANITY CHECK] 데이터 기본 통계")
    print("="*60)

    speeds, gt_x_ends, gt_y_ends, ego_x0s = [], [], [], []
    valid_counts = []

    for feats in records:
        hist = feats["agent_tensor"][0]         # [11, 6]  ego history
        gt   = feats["gt_trajectory"]            # [80, 2]
        gv   = feats["gt_valid"]                 # [80]

        # 마지막 프레임 t=10은 항상 (0,0)이어야 함
        ego_x0s.append(hist[-1, 0])             # should be ~0

        # ego 속도 (ego frame, t=10)
        speed = np.hypot(hist[-1, 2], hist[-1, 3])
        speeds.append(speed)

        # GT: 마지막 유효 위치
        vidx = np.where(gv)[0]
        valid_counts.append(len(vidx))
        if len(vidx):
            gt_x_ends.append(gt[vidx[-1], 0])
            gt_y_ends.append(gt[vidx[-1], 1])

    speeds      = np.array(speeds)
    gt_x_ends   = np.array(gt_x_ends)
    gt_y_ends   = np.array(gt_y_ends)
    ego_x0s     = np.array(ego_x0s)
    valid_counts= np.array(valid_counts)

    print(f"  ego_tensor[-1,0] (현재 x, should be ~0): "
          f"mean={ego_x0s.mean():.4f}  max={np.abs(ego_x0s).max():.4f}")
    print(f"  ego speed at t0 : mean={speeds.mean():.2f} m/s  "
          f"min={speeds.min():.2f}  max={speeds.max():.2f}")
    print(f"  GT valid frames : mean={valid_counts.mean():.1f}/80  "
          f"min={valid_counts.min()}  max={valid_counts.max()}")
    print(f"  GT 최종 x (forward): mean={gt_x_ends.mean():.2f} m  "
          f"std={gt_x_ends.std():.2f}")
    print(f"  GT 최종 y (lateral): mean={gt_y_ends.mean():.2f} m  "
          f"std={gt_y_ends.std():.2f}")

    # 히스토리 패턴: t=0..10의 x 평균
    hist_x_means = []
    for feats in records[:50]:
        hist = feats["agent_tensor"][0, :, 0]   # [11] x values
        hist_x_means.append(hist)
    hist_x_arr = np.array(hist_x_means)  # [50, 11]
    print(f"\n  [ego history x, avg over {len(hist_x_arr)} scenarios]")
    print(f"  t0(현재)={hist_x_arr[:,-1].mean():.2f}  "
          f"t-1={hist_x_arr[:,-2].mean():.2f}  "
          f"t-5={hist_x_arr[:,-6].mean():.2f}  "
          f"t-10={hist_x_arr[:,0].mean():.2f}")
    print("  (앞으로 달리면 t-10이 음수여야 정상)")

    # CV 베이스라인 quick check
    cv = ConstantVelocityBaseline()
    cv_ades = []
    for feats in records[:100]:
        hist = feats["agent_tensor"][0]
        pred = cv.predict(hist)                  # [80,2]
        gt   = feats["gt_trajectory"]
        gv   = feats["gt_valid"]
        vidx = np.where(gv)[0]
        if len(vidx):
            ade = np.linalg.norm(pred[vidx] - gt[vidx], axis=1).mean()
            cv_ades.append(ade)
    print(f"\n  [CV baseline] ADE = {np.mean(cv_ades):.2f} m  "
          f"(first 100 scenarios)")
    return speeds, gt_x_ends


# ──────────────────────────────────────────────────────────────────────────────
# 3. LSTM 학습 + loss 추적
# ──────────────────────────────────────────────────────────────────────────────
def train_and_diagnose(records, args):
    print("\n" + "="*60)
    print("[LSTM 학습 진단]")
    print("="*60)

    device = torch.device(args.device)
    model  = LSTMBaseline(input_size=6, hidden_size=64, num_layers=2).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 데이터를 텐서로 미리 준비
    all_ego  = []
    all_gt   = []
    all_vidx = []

    for feats in records:
        ego_xy = torch.from_numpy(feats["agent_tensor"][0])  # [11,6] all features
        gt     = torch.from_numpy(feats["gt_trajectory"])           # [80,2]
        vidx   = torch.from_numpy(feats["gt_valid"].nonzero()[0].astype(np.int64))
        all_ego.append(ego_xy)
        all_gt.append(gt)
        all_vidx.append(vidx)

    print(f"  데이터 준비: {len(all_ego)}개  device={device}")

    # --- 학습 전 예측 확인 ---
    model.eval()
    with torch.no_grad():
        sample_in  = all_ego[0].unsqueeze(0).to(device)    # [1,11,2]
        sample_out = model(sample_in)[0].cpu().numpy()     # [80,2]
        sample_gt  = all_gt[0].numpy()
        sample_vid = all_vidx[0].numpy()
    print(f"\n  [학습 전] sample pred x[0..4]: {sample_out[sample_vid[:5], 0]}")
    print(f"  [학습 전] sample GT   x[0..4]: {sample_gt[sample_vid[:5], 0]}")

    # --- 학습 루프 ---
    epoch_losses = []
    step_losses  = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for i, (ego, gt, vidx) in enumerate(zip(all_ego, all_gt, all_vidx)):
            inp = ego.unsqueeze(0).to(device)        # [1,11,2]
            tgt = gt.to(device)                      # [80,2]

            pred = model(inp)[0]                     # [80,2]
            loss = F.l1_loss(pred[vidx], tgt[vidx])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            n += 1
            if epoch == 1:
                step_losses.append(loss.item())

        avg = epoch_loss / n
        epoch_losses.append(avg)
        print(f"  Epoch {epoch}/{args.epochs}  avg_loss={avg:.4f}")

    # --- 학습 후 예측 확인 ---
    model.eval()
    with torch.no_grad():
        pred_after = model(sample_in)[0].cpu().numpy()
    print(f"\n  [학습 후] sample pred x[0..4]: {pred_after[sample_vid[:5], 0]}")
    print(f"  [학습 후] sample GT   x[0..4]: {sample_gt[sample_vid[:5], 0]}")

    # --- 전체 ADE 평가 ---
    ades_before = []
    ades_after  = []
    model.eval()
    with torch.no_grad():
        for ego, gt, vidx in zip(all_ego, all_gt, all_vidx):
            inp  = ego.unsqueeze(0).to(device)
            pred = model(inp)[0].cpu().numpy()
            gtnp = gt.numpy()
            vinp = vidx.numpy()
            if len(vinp):
                ades_after.append(
                    np.linalg.norm(pred[vinp] - gtnp[vinp], axis=1).mean()
                )
    print(f"\n  [학습 후 전체] ADE = {np.mean(ades_after):.3f} m")

    return model, step_losses, epoch_losses, (sample_in, sample_gt, sample_vid,
                                               sample_out, pred_after)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 시각화
# ──────────────────────────────────────────────────────────────────────────────
def visualize(step_losses, epoch_losses, sample_data, out_path="debug_lstm.png"):
    sample_in, gt, vidx, pred_before, pred_after = sample_data
    ego_hist = sample_in[0].cpu().numpy()   # [11,2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- (1) Loss 곡선 ---
    ax = axes[0]
    ax.plot(step_losses[:500], alpha=0.7, lw=0.8, label="step loss (ep1, first 500)")
    ax.set_xlabel("step")
    ax.set_ylabel("L1 loss")
    ax.set_title("Training Loss (Epoch 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # epoch losses
    ax2 = ax.twinx()
    ax2.plot(range(1, len(epoch_losses)+1), epoch_losses,
             "ro-", lw=2, label="epoch avg")
    ax2.set_ylabel("epoch avg loss", color="red")
    ax2.legend(loc="upper right")

    # --- (2) Sample 예측 (학습 전) ---
    ax = axes[1]
    ax.plot(ego_hist[:, 0], ego_hist[:, 1],
            "k.-", lw=2, label="Ego history")
    ax.scatter(0, 0, color="black", s=80, zorder=5)
    ax.plot(gt[vidx, 0], gt[vidx, 1],
            "g--", lw=2, label="GT future")
    ax.plot(pred_before[vidx, 0], pred_before[vidx, 1],
            "r-", lw=2, label="LSTM (before)")
    ax.set_title("Sample: Before training")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")

    # --- (3) Sample 예측 (학습 후) ---
    ax = axes[2]
    ax.plot(ego_hist[:, 0], ego_hist[:, 1],
            "k.-", lw=2, label="Ego history")
    ax.scatter(0, 0, color="black", s=80, zorder=5)
    ax.plot(gt[vidx, 0], gt[vidx, 1],
            "g--", lw=2, label="GT future")
    ax.plot(pred_after[vidx, 0], pred_after[vidx, 1],
            "b-", lw=2, label="LSTM (after)")
    ax.set_title(f"Sample: After {len(epoch_losses)} epoch(s)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  시각화 저장: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"Device: {args.device}  |  n_scenarios: {args.n_scenarios}  "
          f"|  epochs: {args.epochs}")

    records = collect_data(args.n_scenarios)
    if len(records) < 10:
        print("ERROR: 데이터가 너무 적습니다. tfrecord 경로를 확인하세요.")
        print(f"  경로: {TRAIN_PATHS[0]}")
        return

    speeds, gt_x_ends = sanity_check(records)

    model, step_losses, epoch_losses, sample_data = train_and_diagnose(records, args)

    out = os.path.join(ROOT, "debug_lstm.png")
    visualize(step_losses, epoch_losses, sample_data, out_path=out)

    # ── 추가 진단: 속도 vs 예측 x 비교 ──────────────────────────────────────
    print("\n" + "="*60)
    print("[추가 진단] 속도 그룹별 LSTM ADE")
    print("="*60)
    device = torch.device(args.device)
    model.eval()

    groups = {"정지(<2m/s)": [], "저속(2-10)": [], "고속(>10)": []}
    with torch.no_grad():
        for feats in records:
            hist  = feats["agent_tensor"][0]
            speed = np.hypot(hist[-1, 2], hist[-1, 3])
            ego_xy = torch.from_numpy(hist).unsqueeze(0).to(device)  # all 6 features
            pred   = model(ego_xy)[0].cpu().numpy()
            gt     = feats["gt_trajectory"]
            vidx   = np.where(feats["gt_valid"])[0]
            if len(vidx) == 0:
                continue
            ade = np.linalg.norm(pred[vidx] - gt[vidx], axis=1).mean()
            if speed < 2:
                groups["정지(<2m/s)"].append(ade)
            elif speed < 10:
                groups["저속(2-10)"].append(ade)
            else:
                groups["고속(>10)"].append(ade)

    for name, ades in groups.items():
        if ades:
            print(f"  {name}: n={len(ades)}  ADE={np.mean(ades):.3f} m")
        else:
            print(f"  {name}: 해당 없음")


if __name__ == "__main__":
    main()
