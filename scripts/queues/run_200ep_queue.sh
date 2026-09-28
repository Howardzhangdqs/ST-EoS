#!/usr/bin/env bash
# ============================================================================
# External-validity upgrade queue (external review, pre-ICLR): full standard
# CIFAR-100 recipe — 200 epochs, cosine lr 0.1→0, pad/crop+flip augmentation,
# n=5 seeds × 3 arms (sgd / accurate rank-1 damping rankkx_d1 / sam),
# exact reference ARPACK curvature every 5 epochs.
#
# Extends revision2 stage 5 (100ep, n=3) to the reviewer-requested 200-epoch
# n≥5 head-to-head. Interleaved BY SEED so each complete seed-triple lands
# every ~6.4 h (SGD ≈32 min, rankkx_d1 ≈294 min, SAM ≈56 min at 200 ep).
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt, runs/revision2/queue_200ep_stdout.log
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 1024"

run () {
  local name="$1"; shift
  if [ -f "runs/$name/manifest.json" ]; then
    echo "[skip] $name (manifest exists)"; return 0
  fi
  local t0=$(date +%s)
  echo "[start] $name  $(date '+%F %T')"
  "$PY" scripts/experiments/train_cifar100_resnet18.py --run-name "$name" "$@" \
      > "$LOGDIR/log_$name.txt" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "[done ] $name  rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%F %T')"
}

# ---------------------------------------------------------------------------
STEPS_EP=391          # 50000/128 ≈ 390.6
STEPS_200EP=78200     # 200 × 391
MEAS_EVERY=1955       # every 5 epochs → 40 exact-ARPACK checkpoints
SEEDS="0 1 2 3 4"

echo "=== 200-EPOCH EXTERNAL-VALIDITY QUEUE (n=5, 3 arms; $(date '+%F %T')) ==="
for S in $SEEDS; do
  run "cifar100_r18_sgd_200ep_s$S" --opt sgd --seed $S --augment \
      --schedule cosine --lr-min 0.0 --steps $STEPS_200EP \
      --outcome-every $MEAS_EVERY $DET
  run "cifar100_r18_rankkx_d1_200ep_s$S" --opt rankkx --edit damp --k 1 \
      --plant-N 50 --seed $S --augment --schedule cosine --lr-min 0.0 \
      --steps $STEPS_200EP --outcome-every $MEAS_EVERY $DET
  run "cifar100_r18_sam_200ep_s$S" --opt sam --rho 0.05 --seed $S --augment \
      --schedule cosine --lr-min 0.0 --steps $STEPS_200EP \
      --outcome-every $MEAS_EVERY $DET
done

echo "=== 200-EPOCH QUEUE COMPLETE $(date '+%F %T') ==="
