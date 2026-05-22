"""
TFRecord → .npz 전처리 캐시 생성.

한 번 실행하면 이후 train.py --use_cache로 빠른 학습 가능.
샤드당 하나의 .npz 파일로 저장.

Run:
    python preprocess.py                        # train 50 + val 8 샤드
    python preprocess.py --split train          # train만
    python preprocess.py --split val            # val만
    python preprocess.py --out_dir ./cache      # 저장 위치 지정
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.tfrecord import iter_tfrecords
from src.data.features import extract_features, extract_risk_label, extract_condition_label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",   type=str, default="both", choices=["train", "val", "both"])
    p.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "cache"))
    p.add_argument("--n_train_shards", type=int, default=50)
    p.add_argument("--n_val_shards",   type=int, default=8)
    return p.parse_args()


def process_shard(shard_path, out_path):
    from waymo_open_dataset.protos import scenario_pb2

    agents, scenes, trafs = [], [], []
    gt_trajs, gt_valids, risk_labels, cond_labels, sc_ids = [], [], [], [], []

    pbar = tqdm(iter_tfrecords([shard_path]), desc="scenarios", unit="sc", leave=False)
    for raw_bytes in pbar:
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw_bytes)
        try:
            f  = extract_features(sc)
            rl = extract_risk_label(sc)
            cl = extract_condition_label(sc)
        except Exception:
            continue
        if not f["gt_valid"].any():
            continue

        agents.append(f["agent_tensor"])
        scenes.append(f["scene_tensor"])
        trafs.append(f["traffic_tensor"])
        gt_trajs.append(f["gt_trajectory"])
        gt_valids.append(f["gt_valid"])
        risk_labels.append(rl)
        cond_labels.append(cl)
        sc_ids.append(sc.scenario_id)
        pbar.set_postfix(saved=len(agents))

    if not agents:
        return 0

    np.savez_compressed(
        out_path,
        agents      = np.array(agents,      dtype=np.float32),   # [N,32,11,10]
        scenes      = np.array(scenes,      dtype=np.float32),   # [N,50,10,6]
        trafs       = np.array(trafs,       dtype=np.float32),   # [N,6,1]
        gt_trajs    = np.array(gt_trajs,    dtype=np.float32),   # [N,80,2]
        gt_valids   = np.array(gt_valids,   dtype=bool),         # [N,80]
        risk_labels = np.array(risk_labels, dtype=np.float32),   # [N,3]
        cond_labels = np.array(cond_labels, dtype=np.float32),   # [N,9]
        sc_ids      = np.array(sc_ids),
    )
    return len(agents)


def main():
    args = parse_args()
    data_root = os.path.join(ROOT, "waymo-motion-v1_3_0")

    shards = []
    if args.split in ("train", "both"):
        for i in range(args.n_train_shards):
            src = os.path.join(data_root, "train",
                               f"training.tfrecord-{i:05d}-of-01000")
            dst = os.path.join(args.out_dir, "train", f"shard_{i:05d}.npz")
            shards.append(("train", i, src, dst))

    if args.split in ("val", "both"):
        for i in range(args.n_val_shards):
            src = os.path.join(data_root, "val",
                               f"validation.tfrecord-{i:05d}-of-00150")
            dst = os.path.join(args.out_dir, "val", f"shard_{i:05d}.npz")
            shards.append(("val", i, src, dst))

    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val"),   exist_ok=True)

    total_sc = 0
    t0 = time.time()
    shard_bar = tqdm(shards, desc="shards", unit="shard")
    for split, idx, src, dst in shard_bar:
        shard_bar.set_description(f"[{split}] shard_{idx:05d}")
        if not os.path.exists(src):
            tqdm.write(f"  [SKIP] {src} 없음")
            continue
        if os.path.exists(dst):
            data = np.load(dst)
            n = len(data["agents"])
            tqdm.write(f"  [SKIP] {split}/shard_{idx:05d}  이미 존재 ({n} scenarios)")
            total_sc += n
            shard_bar.set_postfix(total=total_sc)
            continue

        t1 = time.time()
        n = process_shard(src, dst)
        elapsed = time.time() - t1
        total_sc += n
        shard_bar.set_postfix(total=total_sc)
        tqdm.write(f"  [{split}] shard_{idx:05d}  {n:4d} scenarios  {elapsed:.1f}s")

    print(f"\n완료: {total_sc}개 시나리오  총 {time.time()-t0:.1f}s")
    print(f"캐시 위치: {args.out_dir}")


if __name__ == "__main__":
    main()
