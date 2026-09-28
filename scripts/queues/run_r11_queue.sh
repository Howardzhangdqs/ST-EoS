#!/usr/bin/env bash
# ============================================================================
# Review-round-11 experiment queue (ICLR rebuttal follow-up; 10 runs).
#
# (a) τ=1 endpoint top-up to n=5: cifar100_r18_sam_mb_s{3,4} — EXACT
#     sam_mb protocol of run_r5_queue.sh stage A1 (analyze_tau_
#     interpolation.py maps τ=1 → cifar100_r18_sam_mb_s*), only the seed
#     differs.
# (b) Negative-sign linSAM control: cifar100_r18_linsamneg_mb_s{0..4} —
#     linsam_mb protocol VERBATIM (same ρ=0.05, same HVP path/defaults,
#     same mb flags) with the correction sign flipped:
#     g ← g − (ρ/‖g‖)·H_B g  (trainer --opt linsam_neg). Norm-matched to
#     linSAM by construction (sign flip preserves ‖(ρ/‖g‖)H_B g‖) —
#     adjudicates sign-rule failure vs weak low-rank actuation for the
#     stochastic null of the eigenspace arms.
# (c) Canonical-protocol checkpoints: cifar100_r18_sgd_detarpack_mb2c_s{0,1,2}
#     — sgd_detarpack_mb2 protocol verbatim + --save-final (final_model.pt
#     for offline top-100 subspace-share / estimator-m analyses).
#
# Protocol (manifest-verified): bs128, 5000 steps, outcome-every 200,
# multibatch, deterministic ARPACK k=3 tol 1e-6, eval batch 512, HVP chunk
# 1024 (linsam/linsam_neg training-HVP path = legacy single-graph, i.e.
# --linsam-chunk default 0 — resnet18 BatchNorm forbids the chunked train
# path; verified vs run_r5_queue.sh and log_cifar100_r18_linsam_mb_s0.txt).
# Expected wall: sam 2×~27 min + linsamneg 5×~31 min + sgd-mb2c 3×~25-40 min
# ≈ 4.5-5.5 h.
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r11_stdout.log.
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 1024"
COMMON="--steps 5000 --outcome-every 200 --multibatch $DET"

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
echo "=== (a) SAM τ=1 endpoint top-up seeds 3-4 (n=3→5) ($(date '+%F %T')) ==="
for S in 3 4; do
  run "cifar100_r18_sam_mb_s$S" --opt sam --rho 0.05 --seed $S $COMMON
done

echo "=== (b) negative-sign linSAM, norm-matched control n=5 ($(date '+%F %T')) ==="
for S in 0 1 2 3 4; do
  run "cifar100_r18_linsamneg_mb_s$S" --opt linsam_neg --rho 0.05 --seed $S $COMMON
done

echo "=== (c) sgd mb2 protocol + --save-final checkpoints ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_sgd_detarpack_mb2c_s$S" --opt sgd --seed $S --save-final $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
