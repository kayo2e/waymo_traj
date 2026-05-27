"""
빠른 검증: 두 모델이 과거 궤적(속도)를 실제로 활용하는지 수치로 확인.

테스트:
  1. 속도 일치도 — ego 속도를 0 / 5 / 10 m/s 로 바꿔 예측 거리가 비례하는지
  2. 속도 상관계수 — 속도와 1초 예측 거리의 Pearson r
  3. 방향 일치도 — ego 진행 방향 vs 예측 방향 코사인 유사도

사용법:
  python quick_validate.py --ckpt checkpoints/model_best.pt --label baseline
  python quick_validate.py --ckpt checkpoints/traj_fix/model_best.pt --label traj_fix
"""

import argparse, os, sys
import numpy as np
import torch

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.models.motion_model import RiskConditionedModel


def make_scene(speed: float, heading_deg: float = 0.0, device="cpu"):
    """ego가 speed(m/s)로 heading 방향으로 이동하는 합성 장면 생성."""
    B = 1
    T = 11
    heading = np.radians(heading_deg)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    vx = speed * cos_h
    vy = speed * sin_h
    dt = 0.1  # WOMD 10Hz

    ego = np.zeros((B, T, 10), dtype=np.float32)
    for t in range(T):
        # 과거 위치: 현재(t=10)가 원점 → 과거로 갈수록 음수
        offset = (t - (T - 1)) * dt
        ego[0, t, 0] = offset * cos_h   # x
        ego[0, t, 1] = offset * sin_h   # y
        ego[0, t, 2] = vx               # vx
        ego[0, t, 3] = vy               # vy
        ego[0, t, 4] = cos_h            # cos_h
        ego[0, t, 5] = sin_h            # sin_h
        ego[0, t, 6] = 1.0              # valid_t
        ego[0, t, 7] = t / (T - 1)     # t_norm
        ego[0, t, 8] = 1.0             # type_vehicle

    social = np.zeros((B, 31, T, 10), dtype=np.float32)
    map_sc = np.zeros((B, 50, 10, 6),  dtype=np.float32)
    traf   = np.zeros((B, 6,  1),      dtype=np.float32)
    risk   = np.zeros((B, 3),          dtype=np.float32)

    return (
        torch.from_numpy(ego).to(device),
        torch.from_numpy(social).to(device),
        torch.from_numpy(map_sc).to(device),
        torch.from_numpy(traf).to(device),
        torch.from_numpy(risk).to(device),
    )


def evaluate_model(model, device, n_heading=8):
    """다양한 속도·방향 조합으로 속도 일치도 측정."""
    speeds   = [0.0, 2.0, 5.0, 10.0, 15.0]
    headings = np.linspace(0, 360, n_heading, endpoint=False)

    speed_list, pred_dist_list, dir_cos_list = [], [], []

    model.eval()
    with torch.no_grad():
        for speed in speeds:
            for hdg in headings:
                ego, soc, mp, traf, risk = make_scene(speed, hdg, device)
                out = model(ego, soc, mp, traf, risk_label=risk)
                traj = out["trajectory"][0]  # [K, 80, 2]

                # 1초(10 steps) 예측 — best-mode: 원점에서 가장 먼 거리
                pos_1s = traj[:, 9, :]           # [K, 2]
                dist_1s = pos_1s.norm(dim=-1).max().item()

                # 방향 코사인: ego heading vs best-mode 예측 방향
                best_k = pos_1s.norm(dim=-1).argmax()
                pred_dir = pos_1s[best_k]
                pred_norm = pred_dir.norm().item()
                if speed > 0 and pred_norm > 1e-3:
                    ego_dir = torch.tensor(
                        [np.cos(np.radians(hdg)), np.sin(np.radians(hdg))],
                        dtype=torch.float32, device=device
                    )
                    cos_sim = (pred_dir / pred_norm).dot(ego_dir / ego_dir.norm()).item()
                    dir_cos_list.append(cos_sim)

                speed_list.append(speed)
                pred_dist_list.append(dist_1s)

    speed_arr = np.array(speed_list)
    dist_arr  = np.array(pred_dist_list)

    # 속도-거리 상관계수
    mask = speed_arr > 0
    r = np.corrcoef(speed_arr[mask], dist_arr[mask])[0, 1] if mask.sum() > 1 else 0.0

    # 속도별 평균 예측 거리
    speed_means = {}
    for sp in speeds:
        idx = speed_arr == sp
        speed_means[sp] = dist_arr[idx].mean()

    dir_cos_mean = np.mean(dir_cos_list) if dir_cos_list else 0.0

    return speed_means, r, dir_cos_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  type=str, default=None)
    p.add_argument("--label", type=str, default="model")
    p.add_argument("--device",type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_lane_mamba", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    model  = RiskConditionedModel(d_model=128, K=6, n_layers=2,
                                  use_lane_mamba=not args.no_lane_mamba).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        ep  = ckpt.get("epoch", "?")
        ade = ckpt.get("ev_ade", float("nan"))
        print(f"[{args.label}] epoch={ep}  val_minADE={ade:.3f}m")
    else:
        print(f"[{args.label}] random init (no checkpoint)")

    speed_means, r, dir_cos = evaluate_model(model, device)

    print(f"\n{'='*55}")
    print(f"  [{args.label}] 속도-예측 일치도 테스트")
    print(f"{'='*55}")
    print(f"  {'속도(m/s)':<12} {'1초 예측거리(m)':>16}")
    print(f"  {'-'*30}")
    for sp, d in speed_means.items():
        expected = sp * 1.0  # 1초 × 속도 = 예상 거리
        print(f"  {sp:<12.1f} {d:>12.3f}m   (기대: {expected:.1f}m)")

    print(f"\n  속도↔거리 상관계수 r = {r:.4f}  (1.0이 이상적)")
    print(f"  방향 코사인 유사도    = {dir_cos:.4f}  (1.0이 이상적)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
