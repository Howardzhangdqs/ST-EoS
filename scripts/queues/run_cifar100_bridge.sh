#!/usr/bin/env bash
# Driver: bridge experiment + deterministic CIFAR-100 three-way re-validation.
#
# Bridge (ora-1's top recommendation): clamp-free C3 plant on ResNet-18 — does
# removing the self-gating clamp make rank-1 (a) stay ≈SGD (real batch-sharpness
# phenomenon, clamp irrelevant) or (b) sharp-escape like the MLP accurate-v_top
# (the clamp was hiding it)? This resolves whether the C3-at-scale inertness is
# a clamp artifact or genuine, and reconciles the two rank-1 plants (C3 vs X-SAM).
#
# Deterministic three-way: re-runs sgd/rank1(clamped)/sam in deterministic CUDA
# so the "non-overlapping CI" scale claim is on the same footing as the X-SAM
# comparison (default CUDA has ~6-unit within-seed non-determinism).
#
# 4 opts × seed, deterministic: sgd / rank1 (clamped) / rank1_noclamp (bridge) / sam.
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS

SEED="${1:?usage: run_cifar100_bridge.sh <seed>}"
STEPS=5000
LOG="runs/log_cifar100_bridge${SEED}.txt"

echo "========================================" | tee "$LOG"
echo "CIFAR-100/ResNet-18 bridge + det 3-way — seed $SEED (deterministic)" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run () {  # run <run-suffix> <opt> <extra-args...>
    local SUFFIX="$1"; local OPT="$2"; shift 2
    local RN="cifar100_r18_${SUFFIX}_s${SEED}"
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === opt=$OPT (suffix=$SUFFIX) seed=$SEED (deterministic) ===" | tee -a "$LOG"
    python scripts/experiments/train_cifar100_resnet18.py \
        --dataset cifar100 --opt "$OPT" --steps "$STEPS" \
        --sharp-every 100 --sharp-niter 200 \
        --deterministic "$@" \
        --seed "$SEED" --run-name "$RN" \
        2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] === $SUFFIX seed=$SEED done ===" | tee -a "$LOG"
}

run sgd sgd
run rank1 rank1                 # clamped C3 (deterministic 3-way re-validation)
run rank1_noclamp rank1 --no-clamp   # clamp-free C3 (bridge) — distinct run-name
run sam sam

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL DONE seed=$SEED: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
