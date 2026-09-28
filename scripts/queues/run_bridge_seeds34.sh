#!/usr/bin/env bash
# Driver (reviewer item 2): clamp-free bridge SEED SCALING — seeds 3 and 4 of
# the EXACT 4-arm bridge protocol from run_cifar100_bridge.sh (sgd / rank1
# clamped / rank1_noclamp / sam, deterministic, 5000 steps, sharp-every 100,
# sharp-niter 200), extending the existing n=3 (seeds 0-2) to n=5. Run names
# are unchanged (cifar100_r18_{suffix}_s{seed}) so downstream aggregation
# (analyze scripts reading runs/cifar100_r18_*_{s0..4}) picks them up
# uniformly. Sequential; || exit 1 on any failure.
set -e
set -o pipefail
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS

SEEDS="3 4"
STEPS=5000

for SEED in $SEEDS; do
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
            2>&1 | tee -a "$LOG" || exit 1
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
done

echo "" | tee -a "$LOG"
echo "BRIDGE SEEDS 3+4 ALL DONE: $(date)"
