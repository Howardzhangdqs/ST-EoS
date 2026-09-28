#!/usr/bin/env bash
# Re-run the clamp-free bridge sweep (4-way x 3-seed, CIFAR-100/ResNet-18) with
# FINE-GRAINED sharpness measurement: --sharp-every 20 (was 100) so the
# fig:bridge trajectory panel (32_bridge_clampfree_c3.pdf) gets 251 pts/curve.
# Protocol otherwise IDENTICAL to run_cifar100_bridge.sh (same power-iter
# λ̂_g proxy, sharp-niter 200, deterministic CUDA). Old every-100 metrics are
# backed up to metrics_every100.csv (one-time) before overwrite.
# Usage: run_cifar100_bridge_d20.sh <seed>
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

SEED="${1:?usage: run_cifar100_bridge_d20.sh <seed>}"
STEPS=5000
SHARP_EVERY=20
LOG="runs/log_cifar100_bridge_d20_${SEED}.txt"

echo "========================================" | tee "$LOG"
echo "CIFAR-100 bridge FINE (every-20 sharpness) — seed $SEED (deterministic)" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run () {  # run <run-suffix> <opt> <extra-args...>
    local SUFFIX="$1"; local OPT="$2"; shift 2
    local RN="cifar100_r18_${SUFFIX}_s${SEED}"
    if [ -f "runs/${RN}/metrics.csv" ] && [ ! -f "runs/${RN}/metrics_every100.csv" ]; then
        cp "runs/${RN}/metrics.csv" "runs/${RN}/metrics_every100.csv"
    fi
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === opt=$OPT (suffix=$SUFFIX) seed=$SEED (every-20) ===" | tee -a "$LOG"
    python scripts/experiments/train_cifar100_resnet18.py \
        --dataset cifar100 --opt "$OPT" --steps "$STEPS" \
        --sharp-every "$SHARP_EVERY" --sharp-niter 200 \
        --deterministic "$@" \
        --seed "$SEED" --run-name "$RN" \
        2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === $SUFFIX seed=$SEED done ===" | tee -a "$LOG"
}

run sgd sgd
run rank1 rank1                 # clamped C3
run rank1_noclamp rank1 --no-clamp   # clamp-free C3 (bridge)
run sam sam

echo "" | tee -a "$LOG"
echo "ALL DONE seed $SEED (every-20): $(date)" | tee -a "$LOG"
