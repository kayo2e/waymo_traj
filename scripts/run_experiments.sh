#!/bin/bash
set -e

PREPROCESS_PID=2473332
LOG_DIR="/home/dtlab/gy/waymo_traj/waymo_traj"

echo "[$(date '+%H:%M:%S')] 전처리 완료 대기 중 (PID: $PREPROCESS_PID)..."

while kill -0 $PREPROCESS_PID 2>/dev/null; do
    DONE=$(ls $LOG_DIR/cache/train/*.npz 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] 진행 중: train $DONE/50 샤드 완료"
    sleep 60
done

echo "[$(date '+%H:%M:%S')] 전처리 완료!"
DONE=$(ls $LOG_DIR/cache/train/*.npz 2>/dev/null | wc -l)
VAL_DONE=$(ls $LOG_DIR/cache/val/*.npz 2>/dev/null | wc -l)
echo "  train: $DONE 샤드, val: $VAL_DONE 샤드"

echo ""
echo "===== [1/2] RISK conditioning 학습 시작 ====="
echo "[$(date '+%H:%M:%S')] run_name: traj_risk"
conda run -n waymo2 python $LOG_DIR/train.py \
    --run_name traj_risk \
    --use_cache \
    --cond_type risk \
    --epochs 10 \
    --lr 1e-4 \
    --loss laplace \
    --device cuda \
    2>&1 | tee $LOG_DIR/log_traj_risk.txt

echo ""
echo "===== [2/2] MANEUVER conditioning 학습 시작 ====="
echo "[$(date '+%H:%M:%S')] run_name: traj_maneuver"
conda run -n waymo2 python $LOG_DIR/train.py \
    --run_name traj_maneuver \
    --use_cache \
    --cond_type maneuver \
    --epochs 10 \
    --lr 1e-4 \
    --loss laplace \
    --device cuda \
    2>&1 | tee $LOG_DIR/log_traj_maneuver.txt

echo ""
echo "===== 모든 실험 완료 ====="
echo "[$(date '+%H:%M:%S')] 결과:"
echo "  - log_traj_risk.txt"
echo "  - log_traj_maneuver.txt"
