"""
학습 곡선 비교 시각화.

자동으로 checkpoints/ 하위 모든 run을 스캔하고
minADE / minFDE / MR 학습 곡선을 그립니다.

사용법:
    python plot_curves.py                          # 전체 run 비교
    python plot_curves.py --runs baseline_quick traj_gpt_anchor
    python plot_curves.py --out curves.png --dpi 150
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

ROOT     = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# run별 표시 이름 / 색 / 스타일 매핑 (없으면 자동)
STYLE_MAP = {
    "baseline_quick":   dict(label="Baseline (mean-pool)",           color="#4C72B0", ls="-",  marker="o"),
    "traj_gpt_quick":   dict(label="TrajGPT (absolute pred)",        color="#DD8452", ls="--", marker="s"),
    "traj_gpt_delta":   dict(label="TrajGPT (delta pred)",           color="#C44E52", ls=":",  marker="^"),
    "traj_gpt_anchor":  dict(label="TrajGPT (anchor+causal) ★",     color="#55A868", ls="-",  marker="D"),
    "traj_fix":         dict(label="RCM + time_emb + vel_proj ★",    color="#8172B2", ls="-",  marker="v"),
}

COLORS  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
MARKERS = ["o","s","D","^","v","P","X","*","h","<"]


def load_run(run_dir: str):
    """checkpoint 디렉토리에서 epoch별 메트릭 추출."""
    records = []
    for fname in sorted(os.listdir(run_dir)):
        if not fname.startswith("model_epoch") or not fname.endswith(".pt"):
            continue
        path = os.path.join(run_dir, fname)
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        ep = ckpt.get("epoch")
        if ep is None:
            continue
        records.append({
            "epoch":  ep,
            "tr_ade": ckpt.get("tr_ade", float("nan")),
            "tr_fde": ckpt.get("tr_fde", float("nan")),
            "tr_mr":  ckpt.get("tr_mr",  float("nan")),
            "ev_ade": ckpt.get("ev_ade", float("nan")),
            "ev_fde": ckpt.get("ev_fde", float("nan")),
            "ev_mr":  ckpt.get("ev_mr",  float("nan")),
        })
    records.sort(key=lambda x: x["epoch"])
    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",  nargs="*", default=None,
                   help="비교할 run 이름 (생략 시 전체 자동 스캔)")
    p.add_argument("--out",   type=str, default="curves.png")
    p.add_argument("--dpi",   type=int, default=150)
    p.add_argument("--title", type=str, default="Training Curve Comparison  (val split, max_scenarios=1000)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── run 목록 결정 ────────────────────────────────────────────────────────
    if args.runs:
        run_names = args.runs
    else:
        run_names = sorted([
            d for d in os.listdir(CKPT_DIR)
            if os.path.isdir(os.path.join(CKPT_DIR, d))
        ])

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    runs = {}
    for name in run_names:
        path = os.path.join(CKPT_DIR, name)
        if not os.path.isdir(path):
            print(f"[skip] {name} — 디렉토리 없음")
            continue
        recs = load_run(path)
        if not recs:
            print(f"[skip] {name} — epoch 체크포인트 없음")
            continue
        runs[name] = recs
        print(f"[load] {name}: {len(recs)} epochs  "
              f"best_ev_ade={min(r['ev_ade'] for r in recs):.3f}m")

    if not runs:
        print("표시할 데이터가 없습니다.")
        sys.exit(1)

    # ── 그림 구성: 3개 패널 (ADE / FDE / MR) ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(args.title, fontsize=13, fontweight="bold", y=1.01)

    metrics = [
        ("ev_ade", "minADE (m)",      "lower is better"),
        ("ev_fde", "minFDE (m)",      "lower is better"),
        ("ev_mr",  "MR (miss rate)",  "lower is better"),
    ]

    color_cycle  = iter(COLORS)
    marker_cycle = iter(MARKERS)

    for i, (run_name, recs) in enumerate(runs.items()):
        style = STYLE_MAP.get(run_name, {})
        color  = style.get("color",  next(color_cycle))
        ls     = style.get("ls",     "-")
        marker = style.get("marker", next(marker_cycle))
        label  = style.get("label",  run_name)

        epochs = [r["epoch"] for r in recs]

        for ax, (key, ylabel, note) in zip(axes, metrics):
            vals = [r[key] for r in recs]
            ax.plot(epochs, vals,
                    color=color, ls=ls, marker=marker,
                    linewidth=2, markersize=6, label=label)

    for ax, (key, ylabel, note) in zip(axes, metrics):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}\n({note})", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # y축 상한: 이상치 제거용 (최솟값 × 6 이하만 표시)
        all_vals = []
        for recs in runs.values():
            all_vals += [r[key] for r in recs if not __import__('math').isnan(r[key])]
        if all_vals:
            lo = min(all_vals)
            ax.set_ylim(bottom=max(0, lo * 0.9),
                        top=min(max(all_vals), lo * 6 + 0.5))

    plt.tight_layout()
    out_path = os.path.join(ROOT, args.out)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # ── 텍스트 요약표 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Run':<25} {'ep':>3}  {'ADE':>7}  {'FDE':>7}  {'MR':>6}")
    print("  " + "-" * 56)
    for run_name, recs in runs.items():
        label = STYLE_MAP.get(run_name, {}).get("label", run_name)
        best  = min(recs, key=lambda r: r["ev_ade"])
        print(f"  {label:<25} {best['epoch']:>3}  "
              f"{best['ev_ade']:>7.3f}  {best['ev_fde']:>7.3f}  {best['ev_mr']:>6.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
