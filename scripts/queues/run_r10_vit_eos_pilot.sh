#!/usr/bin/env bash
# ============================================================================
# Review-round-10 pilot — ViT-tiny EoS operating-point search (lr grid).
#
# Goal (pre-registered): find the lr at which vit_tiny enters the EoS regime
# within the canonical 5000-step deterministic protocol, for the phase-2
# 4-arm control (sgd / rankkx_d1 / linsam / sam) at that operating point.
#
# Reference point (r8): lr=0.01 -> 2/eta=200, late lambda_max ~ 40.4+-2.8
# (SGD) — deep sub-EoS (ratio ~ 0.2).
#
# Grid (sgd only, seed 0): lr in {0.02, 0.05, 0.1, 0.2} -> 2/eta in
# {100, 40, 20, 10}.
#
# Selection criterion for phase 2 (fixed before any pilot run completed):
#   late window = outcomes with step >= 2500;
#   pick the lr whose median late lambda_max/(2/eta) lies in [0.8, 1.3],
#   with no NaN steps, non-divergent loss, and a lambda_max trajectory that
#   flattens/oscillates (not still monotonically climbing at run end).
#   Tie-break: the lr closest to ratio 1.0.
#
# Protocol: identical to r8 (deterministic CUDA, ARPACK k=3 tol 1e-6 on the
# 512-example eval batch, chunk-size 128 — the ViT SDPA memory cap, see r8
# header), bs 128, constant lr (schedule none), outcome every 200 steps.
#
# Expected cost: 4 x ~16–20 min = ~1.2–1.5 h (GPU uncontended).
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r10_pilot_stdout.log.
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 128"
COMMON="--model vit_tiny --steps 5000 --outcome-every 200 $DET"

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

for LR in 0.02 0.05 0.1 0.2; do
  TAG=$(echo "$LR" | tr -d '.')   # 002 / 005 / 01 / 02
  run "cifar100_vittiny_eoslr${TAG}_sgd_s0" --opt sgd --seed 0 --lr "$LR" $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
