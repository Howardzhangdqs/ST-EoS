#!/usr/bin/env bash
# Driver: run SGD / rank1 / SAM for one seed on CIFAR-100 / ResNet-18.
# Runs sequentially inside the tmux session so the GPU is never contended.
# Logs everything to runs/log_seed{SEED}.txt for tailing.
set -e
cd "$(dirname "$0")/../.."

SEED="${1:?usage: run_cifar100_seed.sh <seed>}"
STEPS=5000
LOG="runs/log_seed${SEED}.txt"

echo "========================================" | tee "$LOG"
echo "CIFAR-100 / ResNet-18 control — seed $SEED" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for OPT in sgd rank1 sam; do
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED steps=$STEPS ===" | tee -a "$LOG"
    python scripts/experiments/train_cifar100_resnet18.py \
        --opt "$OPT" --steps "$STEPS" \
        --sharp-every 100 --sharp-niter 80 \
        --seed "$SEED" \
        --run-name "cifar100_r18_${OPT}_s${SEED}" \
        2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL DONE seed=$SEED: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
