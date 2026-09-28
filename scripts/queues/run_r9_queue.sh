#!/usr/bin/env bash
# ============================================================================
# Review-round-9 experiment queue — τ-interpolation of SAM (2 τ × 5 seeds).
#
# sam_interp steps with g ← (1−τ)·g_lin + τ·g_SAM at fixed ρ=0.05: τ=1 is
# bit-identical to the sam arm, τ=0 to linsam_bnmatched (both endpoint
# contracts verified bit-exact by scripts/analysis/validate_tau_match.py on
# 2026-09-12, 60-step deterministic runs, 17 numeric columns max rel diff
# 0.0e+00). Sweeping τ ∈ {0.25, 0.5} traces how much of SAM's λ_max
# reduction survives as the nonlinear perturb-and-requery content is
# dialled out — the interpolation counterpart of the round-5 ρ-sweep.
#
# Protocol: EXACT mb-protocol flags of the sam_mb/linsam_mb baselines and
# the round-7 BN controls (bs128, 5000 steps, outcome-every 200, multibatch,
# deterministic ARPACK, eval batch 512) → directly paired analysis.
# Expected per-run wall ≈ linsam_bnmatched ≈ 31 min → 10 runs ≈ 5.2 h.
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r9_stdout.log.
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
# Seed-interleaved: one τ-pair (~62 min) per iteration.
for S in 0 1 2 3 4; do
  echo "=== SEED $S τ-pair ($(date '+%F %T')) ==="
  run "cifar100_r18_sam_interp_t0.25_mb_s$S" --opt sam_interp --tau 0.25 --rho 0.05 --seed $S $COMMON
  run "cifar100_r18_sam_interp_t0.5_mb_s$S"  --opt sam_interp --tau 0.5  --rho 0.05 --seed $S $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
