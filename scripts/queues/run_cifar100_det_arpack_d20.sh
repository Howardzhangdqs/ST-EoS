#!/usr/bin/env bash
# Dense-sampling variant of the deterministic ARPACK three-way (Fig 30 a/b):
# IDENTICAL protocol to run_cifar100_det_arpack_threeway.sh except
# OUTCOME_EVERY=20 (251 points per trajectory instead of 51) so the paper
# figure curves are smooth. Run-names use _detarpack_d20 so neither the
# canonical _detarpack runs nor the legacy non-deterministic runs are touched.
#
# Determinism guarantee: same seeds + deterministic CUDA => training
# trajectories are bit-identical to the _detarpack runs; lambda_max at
# overlapping steps (multiples of 100) must reconcile against them.
#
# Time: 15 runs, ARPACK ~5x more frequent than the every-100 protocol
# (which took 15-31 min/run) -> expect roughly 1-1.5 h/run, ~15-20 h total,
# sequential (no GPU contention).
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS
mkdir -p runs/main

OPTS=(sgd rank1 sam)
SEEDS=(0 1 2 3 4)
STEPS=5000
EVAL_BATCH=512          # P0 B_sat=512 (full-batch lambda_max within 5% of B=51200)
CHUNK=1024              # chunked-HVP cap; fits 16 GB GPU
OUTCOME_EVERY=20        # DENSE: 251 ARPACK measurements per run (vs 51)

LOG=runs/main/cifar100_det_arpack_d20.log
echo "========================================" | tee "$LOG"
echo "CIFAR-100/ResNet-18 det ARPACK three-way DENSE (every $OUTCOME_EVERY) — $(date)" | tee -a "$LOG"
echo "opts: ${OPTS[*]} | seeds: ${SEEDS[*]} | steps=$STEPS | outcome-every=$OUTCOME_EVERY" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for SEED in "${SEEDS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        # resume-safety: skip runs whose manifest already exists
        if [ -f "runs/cifar100_r18_${OPT}_detarpack_d20_s${SEED}/manifest.json" ]; then
            echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED already done, skip ===" | tee -a "$LOG"
            continue
        fi
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED (dense d20) ===" | tee -a "$LOG"
        PYTHONUNBUFFERED=1 python -u scripts/experiments/train_cifar100_resnet18.py \
            --dataset cifar100 --opt "$OPT" --steps "$STEPS" \
            --deterministic \
            --outcome-arpack --outcome-every "$OUTCOME_EVERY" \
            --outcome-tol 1e-6 --outcome-k 3 \
            --eval-batch "$EVAL_BATCH" --chunk-size "$CHUNK" \
            --seed "$SEED" \
            --run-name "cifar100_r18_${OPT}_detarpack_d20_s${SEED}" \
            >> "runs/main/detarpack_d20_${OPT}_s${SEED}.log" 2>&1
        echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
