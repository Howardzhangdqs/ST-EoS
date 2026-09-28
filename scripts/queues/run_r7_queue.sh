#!/usr/bin/env bash
# ============================================================================
# Review-round-7 experiment queue — BN-protocol controls (3 arms × 5 seeds).
#
# The reference λ_max is measured in EVAL mode, i.e. through the BN RUNNING
# statistics. sam_step updates those stats TWICE per step (forwards at θ and
# θ+ε), plain sgd/linsam ONCE — so part of SAM's measured λ_max reduction
# could in principle be BN-state evolution, not the optimizer. Three controls
# isolate it (all bit-verified against sam_step's buffer evolution by
# scripts/analysis/validate_bn_match.py on 2026-09-12):
#
#   sam_bnfreeze      official SAM BN protocol (disable_running_stats):
#                     training dynamics identical to sam, running stats FROZEN.
#   sgd_bn2x          pure-SGD update + sam's SECOND forward reproduced exactly
#                     (θ+ρ·g/‖g‖, same batch, no autograd): BN evolves as in
#                     sam, update stays SGD.
#   linsam_bnmatched  linSAM's first-order gradient + the same dummy second
#                     forward: BN evolves as in sam, update stays linearized.
#
# Protocol: EXACT mb-protocol flags of the sam_mb/linsam_mb baselines
# (bs128, 5000 steps, outcome-every 200, multibatch, deterministic ARPACK,
# eval batch 512) → directly paired against cifar100_r18_{sam,linsam}_mb_s*
# and the sgd baseline cifar100_r18_sgd_detarpack_mb2_s*.
# Seeds interleaved (r6 convention): each seed triple (~86 min) completes a
# partial n=1 analysis unit.
# Expected per-run wall: sam_bnfreeze ≈27 min, sgd_bn2x ≈28 min,
# linsam_bnmatched ≈31 min  → 15 runs ≈ 21.5 h.
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r7_stdout.log.
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
# Seed-interleaved: one BN-protocol triple per iteration.
for S in 0 1 2 3 4; do
  echo "=== SEED $S triple ($(date '+%F %T')) ==="
  run "cifar100_r18_sam_bnfreeze_mb_s$S"     --opt sam_bnfreeze     --rho 0.05 --seed $S $COMMON
  run "cifar100_r18_sgd_bn2x_mb_s$S"         --opt sgd_bn2x         --rho 0.05 --seed $S $COMMON
  run "cifar100_r18_linsam_bnmatched_mb_s$S" --opt linsam_bnmatched --rho 0.05 --seed $S $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
