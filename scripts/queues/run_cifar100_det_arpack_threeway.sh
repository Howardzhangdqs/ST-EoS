#!/usr/bin/env bash
# Deterministic ARPACK three-way re-run (HIGH Issue 4 fix): SGD / rank1 / SAM on
# CIFAR-100 / ResNet-18, n=5 seeds, deterministic CUDA + ARPACK true-λ_max
# outcome measurement. Replaces the Fig-30 headline numbers (biased power-iter on
# non-deterministic CUDA, +45% underestimate per P0) with a single consistent
# protocol matching the X-SAM (n=10) and true_lmax (n=5) runs.
#
# Run-names use a _detarpack suffix so the existing non-deterministic
# cifar100_r18_{opt}_s{0..4} runs (the Fig-30 artifact of record) are NOT
# clobbered and remain available for comparison.
#
# Time: ~15 runs (3 opts × 5 seeds) × ~40 min ≈ 10 h, sequential (no GPU contention).
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS
mkdir -p runs/main

OPTS=(sgd rank1 sam)
SEEDS=(0 1 2 3 4)
STEPS=5000
EVAL_BATCH=512          # P0 B_sat=512 (full-batch λ_max within 5% of B=51200)
CHUNK=1024              # chunked-HVP cap; fits 16 GB GPU
OUTCOME_EVERY=100       # ARPACK outcome measurement period (matches X-SAM density)

LOG=runs/main/cifar100_det_arpack_threeway.log
echo "========================================" | tee "$LOG"
echo "CIFAR-100/ResNet-18 deterministic ARPACK three-way — $(date)" | tee -a "$LOG"
echo "opts: ${OPTS[*]} | seeds: ${SEEDS[*]} | steps=$STEPS | outcome-every=$OUTCOME_EVERY" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for SEED in "${SEEDS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED (deterministic + ARPACK) ===" | tee -a "$LOG"
        PYTHONUNBUFFERED=1 python -u scripts/experiments/train_cifar100_resnet18.py \
            --dataset cifar100 --opt "$OPT" --steps "$STEPS" \
            --deterministic \
            --outcome-arpack --outcome-every "$OUTCOME_EVERY" \
            --outcome-tol 1e-6 --outcome-k 3 \
            --eval-batch "$EVAL_BATCH" --chunk-size "$CHUNK" \
            --seed "$SEED" \
            --run-name "cifar100_r18_${OPT}_detarpack_s${SEED}" \
            >> "runs/main/detarpack_${OPT}_s${SEED}.log" 2>&1
        echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
