#!/usr/bin/env bash
# Driver (reviewer item 3): LONG + data-augmented validation round — SGD /
# rank-1 / SAM on CIFAR-100 / ResNet-18, 12000 steps (~31 epochs at bs=128),
# --augment (replicate-pad-4 + random 32x32 crop + hflip at batch draw time;
# measurement batches stay UN-augmented). Protocol mirrors
# run_cifar100_det_arpack_threeway.sh: deterministic CUDA + ARPACK true-λ_max
# outcome measurement, same eval-batch/chunk/outcome cadence.
#
# Run-names use an _aug12k suffix so existing runs (detarpack, plain) are NOT
# clobbered.
#
# Time: 6 runs (3 opts × 2 seeds) × ~90 min ≈ 9 h, sequential (no GPU contention).
set -e
set -o pipefail
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS
mkdir -p runs/main

OPTS=(sgd rank1 sam)
SEEDS=(0 1)
STEPS=12000               # ~31 epochs (12000 * 128 / 50000)
EVAL_BATCH=512            # P0 B_sat=512 (matches det_arpack_threeway)
CHUNK=1024                # chunked-HVP cap; fits 16 GB GPU
OUTCOME_EVERY=100         # ARPACK outcome cadence (matches det_arpack_threeway)

LOG=runs/main/cifar100_long_aug.log
echo "========================================" | tee "$LOG"
echo "CIFAR-100/ResNet-18 LONG + augmentation round — $(date)" | tee -a "$LOG"
echo "opts: ${OPTS[*]} | seeds: ${SEEDS[*]} | steps=$STEPS (~31 epochs) | augment=ON | outcome-every=$OUTCOME_EVERY" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for SEED in "${SEEDS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED (augment 12k, deterministic + ARPACK) ===" | tee -a "$LOG"
        PYTHONUNBUFFERED=1 python -u scripts/experiments/train_cifar100_resnet18.py \
            --dataset cifar100 --opt "$OPT" --steps "$STEPS" \
            --augment \
            --deterministic \
            --outcome-arpack --outcome-every "$OUTCOME_EVERY" \
            --outcome-tol 1e-6 --outcome-k 3 \
            --eval-batch "$EVAL_BATCH" --chunk-size "$CHUNK" \
            --seed "$SEED" \
            --run-name "cifar100_r18_${OPT}_aug12k_s${SEED}" \
            >> "runs/main/aug12k_${OPT}_s${SEED}.log" 2>&1 || exit 1
        echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
