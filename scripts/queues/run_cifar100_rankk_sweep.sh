#!/usr/bin/env bash
# Driver: rank-k plant-action sweep (k=2,5) on CIFAR-100 / ResNet-18.
# k=1 is already covered by the rank1 runs in run_cifar100_seed.sh
# (rank1 == rankk with --k 1; see train_cifar100_resnet18.py:149-150).
#
# Purpose: reviewer concern #5 — the paper's k-invariance result (k=1≡k=2≡k=5,
# bit-identical) was obtained only on a 1-D-sharp MLP teacher-student landscape.
# ResNet-18 on CIFAR-100 has a heavy-tailed Hessian spectrum (NOT 1-D sharp), so
# this is the real test: does k-invariance BREAK on a genuine CNN? If k>1 gives a
# different lambda_late_mean than k=1, that is the informative result.
#
# Runs sequentially inside the tmux session so the GPU is never contended.
set -e
cd "$(dirname "$0")/../.."

SEEDS="${1:-0 1}"
STEPS=5000
LOG="runs/log_rankk_sweep.txt"

echo "========================================" | tee "$LOG"
echo "CIFAR-100 / ResNet-18 rank-k sweep — seeds $SEEDS" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for SEED in $SEEDS; do
    for K in 2 5; do
        RUN="cifar100_r18_rankk${K}_s${SEED}"
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] === rankk k=$K seed=$SEED steps=$STEPS ===" | tee -a "$LOG"
        python scripts/experiments/train_cifar100_resnet18.py \
            --opt rankk --k "$K" --steps "$STEPS" \
            --sharp-every 100 --sharp-niter 80 \
            --seed "$SEED" \
            --run-name "$RUN" \
            2>&1 | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] === rankk k=$K seed=$SEED done ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "RANK-K SWEEP DONE: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
