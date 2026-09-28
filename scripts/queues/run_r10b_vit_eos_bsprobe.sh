#!/usr/bin/env bash
# ============================================================================
# Review-round-10b probe — ViT-tiny EoS search, batch-size knob.
#
# Background (r10 pilot, 2026-09-14 17:27, 4/4 rc=0): at lr in
# {0.02,0.05,0.1,0.2}, bs 128, the late-window (step>=2500) median
# lambda_max/(2/eta) was 0.34 / 0.34 / 0.41 / 0.39 — criterion [0.8,1.3]
# unmet at every lr, with FLAT late trends (not climbing), and the
# regularity eta*lambda_max_late ~= 0.7–0.8 across a 10x lr range.
# Interpretation: gradient noise holds ViT-tiny at a sub-EoS equilibrium;
# the EoSS-consistent knob is batch size (B -> inf ⇒ lambda_max -> 2/eta;
# ResNet regime-sweep precedent: deterministic response returns at B=2048).
#
# Probe grid (sgd only, seed 0): bs 512 at lr {0.02, 0.05}; bs 2048 at
# lr {0.05, 0.02}. Same deterministic protocol as r10/r8 (ARPACK k=3 tol
# 1e-6, eval batch 512, chunk 128, constant lr, outcome every 200).
#
# Pre-registered decisions (fixed before any probe run):
#   1. Phase-2 operating point = the (B, lr) cell with late-median
#      lambda_max/(2/eta) in [0.8, 1.3] and flat/oscillating late trend;
#      tie-break: closest to 1.0.
#   2. Mechanism check: late ratio increasing monotonically in B at fixed
#      lr confirms the EoSS noise-suppression reading (reportable either
#      way).
#   3. Fallback: if bs 2048 still has late ratio < 0.6, declare
#      ViT-at-EoS infeasible within this protocol family
#      (5000-step x constant-lr x bs<=2048) and reframe the experiment
#      as the sub-EoS equilibrium observation (eta*lambda ~ 0.7-0.8,
#      lr-independent) — a scope statement for the paper, not a phase-2.
#
# Expected cost: bs512 ~32 min/run, bs2048 ~95 min/run → ~4.3 h total.
# Resume-safe: skip a run if its manifest.json exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r10b_stdout.log.
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

run "cifar100_vittiny_eos_b512_lr002_sgd_s0"  --opt sgd --seed 0 --lr 0.02 --bs 512  $COMMON
run "cifar100_vittiny_eos_b512_lr005_sgd_s0"  --opt sgd --seed 0 --lr 0.05 --bs 512  $COMMON
run "cifar100_vittiny_eos_b2048_lr005_sgd_s0" --opt sgd --seed 0 --lr 0.05 --bs 2048 $COMMON
run "cifar100_vittiny_eos_b2048_lr002_sgd_s0" --opt sgd --seed 0 --lr 0.02 --bs 2048 $COMMON

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
