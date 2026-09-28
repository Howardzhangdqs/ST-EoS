#!/usr/bin/env bash
# X-SAM multi-seed (Reviewer #2): 4-way × 3-seed on CIFAR-10/ResNet-18 with
# ARPACK true-λ_max measurement (not just λ̂_g proxy). Re-runs the existing
# seed-0,1 trajectories under identical training but with ARPACK outcome
# measurement, AND adds seed 2. Overwrites the old power-iter metrics.csv with
# one that has BOTH lambda_max (ARPACK) and lambda_hat_g columns.
#
# Protocol matches run_cifar10_xsam.sh (lr=0.1, mom=0.9, wd=5e-4, bs=128,
# 5000 steps, rho=0.05, alpha=0.2) plus ARPACK outcome (B=512, tol=1e-6, k=3)
# every 100 steps. Deterministic throughout. BN save/restore during all HVPs
# (handled inside measure_curvature_arpack).
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
mkdir -p runs/main

OPTS=(sgd sam xsam_faithful xsam_clamped)
SEEDS=(0 1 2)
STEPS=5000
EVAL_BATCH=512
CHUNK=1024
OUTCOME_EVERY=100

echo "========================================" | tee runs/main/xsam_multiseed.log
echo "X-SAM multi-seed (CIFAR-10, ARPACK) — $(date)" | tee -a runs/main/xsam_multiseed.log
echo "opts: ${OPTS[*]} | seeds: ${SEEDS[*]} | steps=$STEPS" | tee -a runs/main/xsam_multiseed.log
echo "========================================" | tee -a runs/main/xsam_multiseed.log

for SEED in "${SEEDS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        echo "" | tee -a runs/main/xsam_multiseed.log
        echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED (deterministic + ARPACK) ===" \
            | tee -a runs/main/xsam_multiseed.log
        # back up old power-iter metrics if present (one-time)
        RUN_DIR="runs/cifar10_r18_${OPT}_s${SEED}"
        if [ -f "$RUN_DIR/metrics.csv" ] && [ ! -f "$RUN_DIR/metrics_poweriter.csv" ]; then
            cp "$RUN_DIR/metrics.csv" "$RUN_DIR/metrics_poweriter.csv"
        fi
        PYTHONUNBUFFERED=1 python -u scripts/experiments/train_cifar100_resnet18.py \
            --dataset cifar10 --opt "$OPT" --steps "$STEPS" \
            --rho 0.05 --alpha 0.2 \
            --deterministic \
            --outcome-arpack --outcome-every "$OUTCOME_EVERY" \
            --outcome-tol 1e-6 --outcome-k 3 \
            --eval-batch "$EVAL_BATCH" --chunk-size "$CHUNK" \
            --seed "$SEED" \
            --run-name "cifar10_r18_${OPT}_s${SEED}" \
            >> "runs/main/xsam_${OPT}_s${SEED}.log" 2>&1
        echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" \
            | tee -a runs/main/xsam_multiseed.log
    done
done

echo "" | tee -a runs/main/xsam_multiseed.log
echo "========================================" | tee -a runs/main/xsam_multiseed.log
echo "ALL DONE: $(date)" | tee -a runs/main/xsam_multiseed.log
echo "========================================" | tee -a runs/main/xsam_multiseed.log
