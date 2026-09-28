#!/usr/bin/env bash
# ============================================================================
# Review-round-8 experiment queue — ViT (timm vit_tiny, LayerNorm) × 4 arms.
#
# Architecture control for the BN confound of round 7: vit_tiny has NO
# running statistics at all (LayerNorm throughout), so any λ_max difference
# between arms there cannot be normalization-state evolution. Arms:
#   sgd          plain SGD + momentum baseline
#   rankkx_d1    rank-1 ARPACK-sensed damping plant (k=1, damp, N=20)
#   linsam       first-order linearized SAM (single forward + HVP)
#   sam          two-forward SAM
#
# Protocol: canonical deterministic 5000-step recipe (outcome-every 200,
# ARPACK k=3 tol 1e-6, eval batch 512) at the ViT lr convention 0.01
# (train_timm / runs/vit_validate precedent; 2/η = 200), bs 128.
# --chunk-size 128 (NOT the ResNet 1024): ViT SDPA-math double-backward on
# the 512 eval batch needs the chunked-HVP memory cap (100-step smoke
# 2026-09-12 OOMed at chunk 1024 in the outcome measurement; chunk 128 is
# algebraically exact for mean-reduction losses, verified same step-0 λ).
#
# 100-step smoke timings (2026-09-12) → 5000-step extrapolation:
#   sgd   0.062 s/step → ~16 min/run   (+26 ARPACK outcomes ≈ 25.9 s each)
#   sam   0.127 s/step → ~22 min/run
#   linsam 0.177 s/step → ~26 min/run
#   rankkx_d1  sgd + ARPACK sense every 20 steps (≈+0.66 s/step avg)
#                                          → ~70 min/run
# Total: 3 seeds × 4 arms ≈ 6.7 h.
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r8_stdout.log.
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 128"
COMMON="--model vit_tiny --lr 0.01 --steps 5000 --outcome-every 200 $DET"

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
# Seed-interleaved: one 4-arm ViT seed unit (~2.2 h) per iteration.
for S in 0 1 2; do
  echo "=== SEED $S unit ($(date '+%F %T')) ==="
  run "cifar100_vittiny_sgd_s$S"       --opt sgd   --seed $S $COMMON
  run "cifar100_vittiny_rankkx_d1_s$S" --opt rankkx --edit damp --k 1 --seed $S $COMMON
  run "cifar100_vittiny_linsam_s$S"    --opt linsam --rho 0.05 --seed $S $COMMON
  run "cifar100_vittiny_sam_s$S"       --opt sam   --rho 0.05 --seed $S $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
