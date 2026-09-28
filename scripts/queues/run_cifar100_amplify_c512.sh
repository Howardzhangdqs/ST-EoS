#!/usr/bin/env bash
# Rank-1 AMPLIFICATION arm at scale (reviewer sign-flip control): CIFAR-100 /
# ResNet-18, deterministic CUDA + ARPACK — the exact protocol of the
# detarpack three-way (run_cifar100_det_arpack_threeway.sh), so the amplify
# arm slots directly into the Fig-30 comparison. Edit: g_eff = g + 0.5·(g·v̂)v̂
# (γ = −0.5 < 0), ARPACK-sensed v̂ every N=20 steps (true_lmax machinery).
#
# Usage: bash scripts/queues/run_cifar100_amplify.sh <seed> [<seed> ...]
#   (pass a subset of seeds per worker for parallelism, e.g. two terminals:
#    bash scripts/queues/run_cifar100_amplify.sh 0 1 2  &  bash scripts/queues/run_cifar100_amplify.sh 3 4)
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
mkdir -p runs/main

ALPHA=0.5
STEPS=5000
EVAL_BATCH=512
CHUNK=512
OUTCOME_EVERY=100

for SEED in "$@"; do
    echo "[$(date +%H:%M:%S)] === amplify α=$ALPHA seed=$SEED (deterministic + ARPACK) ==="
    PYTHONUNBUFFERED=1 python -u scripts/experiments/train_cifar100_resnet18.py \
        --dataset cifar100 --opt amplify --amplify "$ALPHA" --steps "$STEPS" \
        --deterministic \
        --outcome-arpack --outcome-every "$OUTCOME_EVERY" \
        --outcome-tol 1e-6 --outcome-k 3 \
        --eval-batch "$EVAL_BATCH" --chunk-size "$CHUNK" \
        --seed "$SEED" \
        --run-name "cifar100_r18_amp${ALPHA}_detarpack_s${SEED}" \
        >> "runs/main/amp${ALPHA}_detarpack_s${SEED}.log" 2>&1
    echo "[$(date +%H:%M:%S)] === seed $SEED done ==="
done
