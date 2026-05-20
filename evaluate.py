"""
베이스라인 비교 평가 스크립트.

모델 4종 비교:
  1. Constant Velocity      (비학습)
  2. LSTM Baseline          (학습됨, --lstm_ckpt 지정)
  3. RiskConditionedModel   (학습됨, --ckpt 지정)
  4. MTR (SOTA)             (result.pkl 로드, --mtr_result 지정)

NOTE: testing.tfrecord 은 GT 없음 → 평가는 validation 사용.
      MTR result가 있는 시나리오만 4-way 비교, 나머지는 3-way.

Run:
    cd waymo_traj
    python evaluate.py \
        --ckpt      checkpoints/model_best.pt \
        --lstm_ckpt checkpoints/lstm_best.pt \
        [--mtr_result ../MTR/output/waymo/mtr_single_gpu/benchmark/eval/eval_with_train/result.pkl] \
        [--device cuda] [--max_scenarios 500]
"""

import argparse
import os
import pickle
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.tfrecord       import iter_tfrecords
from src.data.features       import extract_features, extract_risk_label
from src.models.motion_model import RiskConditionedModel
from src.models.baselines    import ConstantVelocityBaseline, LSTMBaseline
from src.eval.metrics        import compute_minADE_FDE, compute_MR


# ── 데이터 경로 (val split — test에는 GT 없음) ───────────────────────────────
def _val_shard(n):
    return os.path.join(ROOT, "waymo-motion-v1_3_0", "val",
                        f"validation.tfrecord-{n:05d}-of-00150")

EVAL_PATHS = [_val_shard(i) for i in range(8)]   # val 00000-00007


# ── MTR 유틸 ─────────────────────────────────────────────────────────────────
def load_mtr_results(path):
    """result.pkl → {scenario_id: [item, ...]} 딕셔너리."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = pickle.load(f)
    by_sc = {}
    for sublist in raw:
        for item in sublist:
            sid = item["scenario_id"]
            by_sc.setdefault(sid, []).append(item)
    print(f"[MTR] result.pkl 로드 완료  {len(by_sc)}개 시나리오: {path}")
    return by_sc


def mtr_pred_to_ego(pred_trajs, x0, y0, cos_h, sin_h):
    """MTR global 좌표 예측 → ego-relative 변환.  pred_trajs: (K, T, 2)"""
    dx = pred_trajs[..., 0] - x0
    dy = pred_trajs[..., 1] - y0
    ex =  dx * cos_h + dy * sin_h
    ey = -dx * sin_h + dy * cos_h
    return np.stack([ex, ey], axis=-1)


def get_mtr_pred(sc, mtr_by_id):
    """Scenario protobuf + MTR 딕셔너리 → ego-relative (K,80,2) 또는 None."""
    if mtr_by_id is None:
        return None
    items = mtr_by_id.get(sc.scenario_id)
    if not items:
        return None

    sdc_idx = sc.sdc_track_index
    mtr_item = next(
        (it for it in items if int(it["track_index_to_predict"]) == sdc_idx),
        None,
    )
    if mtr_item is None:
        return None

    t0    = sc.current_time_index
    ego_s = sc.tracks[sdc_idx].states[t0]
    x0, y0 = ego_s.center_x, ego_s.center_y
    cos_h  = float(np.cos(ego_s.heading))
    sin_h  = float(np.sin(ego_s.heading))

    return mtr_pred_to_ego(mtr_item["pred_trajs"], x0, y0, cos_h, sin_h)


# ── 입력 변환 ─────────────────────────────────────────────────────────────────
def _prep_inputs(feats, device):
    """agent_tensor / scene_tensor / traffic_tensor → RiskConditionedModel 입력."""
    agent = feats["agent_tensor"]    # [32, 11, 10]
    scene = feats["scene_tensor"]    # [50, 10, 6]
    traf  = feats["traffic_tensor"]  # [6, 1]

    ego_hist  = agent[0:1]           # [1, 11, 10]
    social    = agent[1:][None]      # [1, 31, 11, 10]
    map_scene = scene[None]          # [1, 50, 10, 6]
    traf_t    = traf[None]           # [1, 6, 1]

    return (
        torch.from_numpy(ego_hist).to(device),
        torch.from_numpy(social).to(device),
        torch.from_numpy(map_scene).to(device),
        torch.from_numpy(traf_t).to(device),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          type=str,
                   default=os.path.join(ROOT, "checkpoints/model_best.pt"))
    p.add_argument("--lstm_ckpt",     type=str,
                   default=os.path.join(ROOT, "checkpoints/lstm_best.pt"))
    p.add_argument("--mtr_result",    type=str,
                   default=os.path.join(
                       ROOT, "..", "MTR", "output", "waymo",
                       "mtr_single_gpu", "benchmark", "eval",
                       "eval_with_train", "result.pkl"),
                   help="MTR result.pkl 경로 (없으면 3-way 비교)")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_scenarios", type=int, default=None)
    return p.parse_args()


def _empty():
    return {"ade": 0.0, "fde": 0.0, "mr": 0.0, "n": 0}


def _update(acc, ade, fde, mr):
    acc["ade"] += ade
    acc["fde"] += fde
    acc["mr"]  += mr
    acc["n"]   += 1


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Eval   : val shards 00000-00007  ({len(EVAL_PATHS)} files)")
    print()

    # ── MTR 결과 로드 ─────────────────────────────────────────────────────────
    mtr_by_id = load_mtr_results(args.mtr_result)
    if mtr_by_id is None:
        print("[MTR] result.pkl 없음 → 3-way 비교 (CV / LSTM / RCM)")

    # ── RiskConditionedModel ──────────────────────────────────────────────────
    mm_model = RiskConditionedModel(d_model=128, K=6, n_layers=2).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        mm_model.load_state_dict(ckpt["model"])
        ev_ade = ckpt.get("ev_ade", float("nan"))
        print(f"[RiskConditionedModel]  체크포인트 로드  "
              f"epoch={ckpt.get('epoch','?')}  val_minADE={ev_ade:.3f}m")
    else:
        print(f"[RiskConditionedModel]  체크포인트 없음 — random init")
    mm_model.eval()

    # ── LSTM Baseline (6-dim 체크포인트와 호환 — 피처 앞 6개만 사용) ──────────
    lstm_model = LSTMBaseline(input_size=6, hidden_size=64, num_layers=2).to(device)
    lstm_trained = False
    if os.path.exists(args.lstm_ckpt):
        try:
            ckpt_l = torch.load(args.lstm_ckpt, map_location=device)
            lstm_model.load_state_dict(ckpt_l["model"])
            lstm_trained = True
            print(f"[LSTM Baseline]     체크포인트 로드  "
                  f"epoch={ckpt_l.get('epoch','?')}  "
                  f"val_ADE={ckpt_l.get('ev_ade',float('nan')):.3f}m")
        except Exception as e:
            print(f"[LSTM Baseline]     체크포인트 로드 실패 ({e}) → random init")
    else:
        print(f"[LSTM Baseline]     체크포인트 없음 — random init")
    lstm_model.eval()

    cv_model   = ConstantVelocityBaseline()
    lstm_label = "LSTM (trained)" if lstm_trained else "LSTM (random)"

    results = {
        "Constant Velocity":        _empty(),
        lstm_label:                 _empty(),
        "RiskConditionedModel K=6": _empty(),
        "MTR (SOTA)":               _empty(),   # n=0 이면 출력 생략
    }

    from waymo_open_dataset.protos import scenario_pb2
    t_start = time.time()
    n_total = 0
    print()

    for raw_bytes in iter_tfrecords(EVAL_PATHS):
        if args.max_scenarios and n_total >= args.max_scenarios:
            break

        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            feats         = extract_features(sc)
            risk_label_np = extract_risk_label(sc)
        except Exception:
            continue
        if not feats["gt_valid"].any():
            continue

        gt_traj  = feats["gt_trajectory"]
        gt_valid = feats["gt_valid"]

        # ── Constant Velocity ─────────────────────────────────────────────────
        cv_pred  = cv_model.predict(feats["agent_tensor"][0])
        ade, fde = compute_minADE_FDE(cv_pred, gt_traj, gt_valid)
        mr       = compute_MR(cv_pred, gt_traj, gt_valid)
        _update(results["Constant Velocity"], ade, fde, mr)

        # ── LSTM (6-dim 슬라이스 — 기존 체크포인트 호환) ──────────────────────
        ego_6d = torch.from_numpy(
            feats["agent_tensor"][0, :, :6]
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            lstm_pred = lstm_model(ego_6d)[0].cpu().numpy()   # [80, 2]
        ade, fde = compute_minADE_FDE(lstm_pred, gt_traj, gt_valid)
        mr       = compute_MR(lstm_pred, gt_traj, gt_valid)
        _update(results[lstm_label], ade, fde, mr)

        # ── RiskConditionedModel ──────────────────────────────────────────────
        ego_hist, social, map_scene, traf = _prep_inputs(feats, device)
        risk_gt = torch.from_numpy(risk_label_np).unsqueeze(0).to(device)
        with torch.no_grad():
            out = mm_model(ego_hist, social, map_scene, traf, risk_label=risk_gt)
        mm_pred  = out["trajectory"][0].cpu().numpy()          # [K, 80, 2]
        ade, fde = compute_minADE_FDE(mm_pred, gt_traj, gt_valid)
        mr       = compute_MR(mm_pred, gt_traj, gt_valid)
        _update(results["RiskConditionedModel K=6"], ade, fde, mr)

        # ── MTR (result.pkl에 있는 경우만) ────────────────────────────────────
        mtr_pred = get_mtr_pred(sc, mtr_by_id)
        if mtr_pred is not None:
            ade, fde = compute_minADE_FDE(mtr_pred, gt_traj, gt_valid)
            mr       = compute_MR(mtr_pred, gt_traj, gt_valid)
            _update(results["MTR (SOTA)"], ade, fde, mr)

        n_total += 1
        if n_total % 200 == 0:
            print(f"  {n_total} scenarios  ({time.time()-t_start:.0f}s)")

    # ── 결과 테이블 ───────────────────────────────────────────────────────────
    n_mtr = results["MTR (SOTA)"]["n"]
    n_rcm = results["RiskConditionedModel K=6"]["n"]
    print(f"\n평가 시나리오: {n_rcm}개 (MTR 매칭: {n_mtr}개)  "
          f"({time.time()-t_start:.0f}s)\n")

    w = 26
    print("=" * 70)
    print("  Baseline Comparison vs WOMD Ground Truth  (val split)")
    print("=" * 70)
    print(f"  {'모델':{w}}  {'minADE (m)':>10}  {'minFDE (m)':>10}  "
          f"{'MR (2m)':>8}  {'n':>6}")
    print(f"  {'-'*w}  {'----------':>10}  {'----------':>10}  "
          f"{'-------':>8}  {'------':>6}")
    for name, acc in results.items():
        n = acc["n"]
        if n == 0:
            continue
        print(f"  {name:{w}}  {acc['ade']/n:>10.3f}  {acc['fde']/n:>10.3f}  "
              f"{acc['mr']/n:>8.3f}  {n:>6}")
    print("=" * 70)
    print()
    print("* MR: Miss Rate — minFDE > 2.0m 비율 (낮을수록 좋음)")
    if n_mtr > 0:
        print(f"* MTR는 result.pkl에 포함된 {n_mtr}개 시나리오 기준")


if __name__ == "__main__":
    main()
