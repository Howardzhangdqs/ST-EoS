#!/usr/bin/env bash
# Driver (reviewer item 2): rank-k sweep SEED SCALING — seeds 2/3/4 of the
# EXACT rankk k∈{2,5} protocol from run_cifar100_rankk_sweep.sh (which ran
# seeds 0/1), extending n=2 → n=5. Same steps/sharpness flags and run-name
# scheme (cifar100_r18_rankk{K}_s{seed}); distinct log so the original sweep
# log is not interleaved. Sequential; || exit 1 on any failure.
set -e
set -o pipefail
cd "$(dirname "$0")/../.."

SEEDS="${1:-2 3 4}"
STEPS=5000
LOG="runs/log_rankk_sweep234.txt"

echo "========================================" | tee "$LOG"
echo "CIFAR-100 / ResNet-18 rank-k sweep (seed scaling) — seeds $SEEDS" | tee -a "$LOG"
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
            2>&1 | tee -a "$LOG" || exit 1
        echo "[$(date +%H:%M:%S)] === rankk k=$K seed=$SEED done ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "RANK-K SWEEP seeds 234 DONE: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
