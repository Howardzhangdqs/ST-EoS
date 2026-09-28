#!/usr/bin/env bash
# Driver: SGD / SAM / X-SAM-faithful / X-SAM-clamped on CIFAR-10 / ResNet-18.
# Resolves reviewer #3 (the X-SAM contradiction). Our F24 found X-SAM's rank-1
# correction inert (0/3000 fires) — but that used a self-gating clamp that real
# X-SAM (Duan et al. 2026, Eq. 11) does NOT have. Real X-SAM is fixed-α=0.2,
# sign-gated, UNCONDITIONAL (fires every step). This driver runs both so the
# λ_max trajectory contrast isolates the clamp as the source of the contradiction.
#
# Setup: ResNet-18 (11.2M), lr=0.1, mom=0.9, wd=5e-4, bs=128, 5000 steps,
# ρ=0.05 (X-SAM's CIFAR-10 radius), α=0.2. Matches our CIFAR-100 control setup
# for cross-dataset comparability (we measure λ_max trajectory, not SOTA acc).
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS

SEED="${1:?usage: run_cifar10_xsam.sh <seed>}"
STEPS=5000
LOG="runs/log_cifar10_xs${SEED}.txt"

echo "========================================" | tee "$LOG"
echo "CIFAR-10 / ResNet-18 X-SAM regime test — seed $SEED (deterministic)" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for OPT in sgd sam xsam_faithful xsam_clamped; do
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED steps=$STEPS (deterministic) ===" | tee -a "$LOG"
    python scripts/experiments/train_cifar100_resnet18.py \
        --dataset cifar10 --opt "$OPT" --steps "$STEPS" \
        --sharp-every 100 --sharp-niter 200 \
        --rho 0.05 --alpha 0.2 \
        --deterministic \
        --seed "$SEED" \
        --run-name "cifar10_r18_${OPT}_s${SEED}" \
        2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL DONE seed=$SEED: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
