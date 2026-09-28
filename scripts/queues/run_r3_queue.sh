#!/usr/bin/env bash
# ============================================================================
# Review-round-3 experiment queue (sequential, single RTX 5080).
#
# Stage A  batch-sharpness measurement : sgd --multibatch rerun (mb2) —
#          identical protocol to revision-2 mb runs (trajectories verified
#          bit-identical) + NEW columns: lhg_curmb (current-minibatch
#          gradient Rayleigh quotient) and lhg_bs_mean/std (m=4-minibatch
#          average = E_B[g^T H_B g/|g|^2], the Andreyev–Beneventano EoSS
#          batch sharpness, distinct from minibatch λmax).
# Stage B  full-rank linearized SAM    : linsam — g ← g + (ρ/‖g‖)H_B g,
#          ρ=0.05 matched to the sam arm, first-order SAM with NO
#          perturbed-point re-query (rank-vs-requery adjudication control).
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt
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

echo "=== STAGE A: batch sharpness (cross-batch study rerun w/ E_B[g'Hg/|g|²]) ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_sgd_detarpack_mb2_s$S" --opt sgd --seed $S --steps 5000 \
      --outcome-every 200 --multibatch $DET
done

echo "=== STAGE B: full-rank linearized SAM (g + (ρ/‖g‖)Hg, ρ=0.05) ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_linsam_s$S" --opt linsam --rho 0.05 --seed $S --steps 5000 \
      --outcome-every 100 $DET
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
