#!/usr/bin/env bash
# Re-run the X-SAM multi-seed sweep (4-way x 10-seed, CIFAR-10/ResNet-18) with
# FINE-GRAINED ARPACK outcome measurement: every 20 steps (was 100) so the
# fig:xsam trajectory panel (28_xsam_ablation.pdf) gets 251 points/curve.
# ARPACK cost config k=1/ncv=8 (benchmarked: 9 matvecs vs 21 at defaults,
# ~2.6x cheaper, rel err ≤1e-7; the figure only consumes lambda_max — the
# every-100 backups keep the k=3 lambda_2 history).
# Protocol otherwise IDENTICAL to run_cifar10_xsam_multiseed.sh (lr=0.1,
# mom=0.9, wd=5e-4, bs=128, 5000 steps, rho=0.05, alpha=0.2, deterministic,
# ARPACK tol=1e-6 on B=512). Determinism => start/end values unchanged;
# only the sampling density increases.
set -e
cd "$(dirname "$0")/../.."

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
mkdir -p runs/main

OPTS=(sgd sam xsam_faithful xsam_clamped)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
STEPS=5000
EVAL_BATCH=512
CHUNK=1024
OUTCOME_EVERY=20
OUTCOME_K=1
OUTCOME_NCV=8

LOG=runs/main/xsam_multiseed_d20.log
echo "========================================" | tee "$LOG"
echo "X-SAM multi-seed FINE (every-20 ARPACK) — $(date)" | tee -a "$LOG"
echo "opts: ${OPTS[*]} | seeds: ${SEEDS[*]} | steps=$STEPS" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for SEED in "${SEEDS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] === opt=$OPT seed=$SEED (every-20 ARPACK) ===" \
            | tee -a "$LOG"
        RUN_DIR="runs/cifar10_r18_${OPT}_s${SEED}"
        # one-time backup of the every-100 ARPACK metrics before overwrite
        if [ -f "$RUN_DIR/metrics.csv" ] && [ ! -f "$RUN_DIR/metrics_every100.csv" ]; then
            cp "$RUN_DIR/metrics.csv" "$RUN_DIR/metrics_every100.csv"
        fi
        PYTHONUNBUFFERED=1 python -u scripts/experiments/train_cifar100_resnet18.py \
            --dataset cifar10 --opt "$OPT" --steps "$STEPS" \
            --rho 0.05 --alpha 0.2 \
            --deterministic \
            --outcome-arpack --outcome-every "$OUTCOME_EVERY" \
            --outcome-tol 1e-6 --outcome-k "$OUTCOME_K" \
            --outcome-ncv "$OUTCOME_NCV" \
            --eval-batch "$EVAL_BATCH" --chunk-size "$CHUNK" \
            --seed "$SEED" \
            --run-name "cifar10_r18_${OPT}_s${SEED}" \
            >> "runs/main/xsam_d20_${OPT}_s${SEED}.log" 2>&1
        echo "[$(date +%H:%M:%S)] === $OPT seed=$SEED done ===" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "ALL DONE (every-20): $(date)" | tee -a "$LOG"

# regenerate the figure + summary from the new dense metrics
python scripts/plots/plot_xsam_multiseed.py 2>&1 | tail -2 | tee -a "$LOG"
echo "figure refreshed: figures/28_xsam_ablation.pdf" | tee -a "$LOG"
