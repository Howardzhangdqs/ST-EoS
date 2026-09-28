#!/usr/bin/env bash
# ============================================================================
# Review-round-10c phase-2 — ViT-tiny 4-arm control AT the EoS operating point.
#
# Operating point (from r10/r10b, pre-registered selection): bs=512, lr=0.05
# (2/eta = 40). r10b measured sgd late-median lambda_max = 38.13, ratio
# 0.95 [IQR 35.4-44.6], oscillating about 2/eta (peak 59.7@800 then band).
# This is the first ViT operating point in the EoS band; r8's point
# (bs128, lr0.01, ratio ~0.2) was deep sub-EoS.
#
# Arms (seed-interleaved, 3 seeds): sgd / rankkx_d1 (ARPACK-sensed rank-1
# damping, k=1, N=20) / linsam (rho=0.05) / sam (rho=0.05). Same
# deterministic protocol as r8/r10 (ARPACK k=3 tol 1e-6, eval batch 512,
# chunk 128, constant lr, outcome every 200, 5000 steps).
#
# Pre-registered predictions/decisions (fixed before launch):
#   P1 protocol: sgd late-median lambda_max/(2/eta) in [0.8, 1.1] across
#      seeds (r10b seed-0 measured 0.95).
#   P2 central question: linsam vs sam at EoS. If linsam ~ sam (as at the
#      below-EoS r8 point) the "first-order suffices on LayerNorm" scope
#      extends into the EoS band; if linsam drifts sharp-ward of SGD (as on
#      BN/ResNet-18) the re-query necessity becomes EoS-regime-dependent
#      rather than architecture-dependent alone. Either outcome integrates
#      as a scope-condition sentence in sub:scale + Limitations.
#   P3 rank-1 damping at EoS: any late-lambda_max movement of rankkx_d1
#      measurably different from SGD breaks the bs128 inertness; per the
#      2-D reduction the expected direction at the boundary is sharp-ward
#      (boundary moves up). Null (indistinguishable at this power) is also
#      reportable. TOST margin: 0.25*|SGD-SAM| (same prespecified family),
#      calibrated wording per FINDINGS 47 convention.
#
# MEMORY GATE for linsam: its HVP is a full-training-batch double-backward
# (no chunking in steos/optim/linsam.py). With the ~8.8 GB of phantom GPU
# memory held by cross-namespace PIDs (6087/74517), linsam at bs512 OOMs
# (smoke-tested 2026-09-14 19:30, rc=1). The queue therefore runs linsam
# ONLY when >= 9000 MiB are free; otherwise it logs [defer] and continues.
# Re-running this script after the phantom memory is freed picks linsam up
# automatically (resume-safe by manifest).
#
# Expected cost: sgd ~27 min, rankkx_d1 ~80 min, sam ~37 min, linsam
# ~43 min (when it runs) → 3 seeds x 3 arms ~7.2 h (+2.2 h with linsam).
# Resume-safe: skip a run if its manifest.json exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r10c_stdout.log.
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 128"
COMMON="--model vit_tiny --lr 0.05 --bs 512 --steps 5000 --outcome-every 200 $DET"

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

run_gated () {  # linsam: needs >= 9000 MiB free (phantom-memory workaround)
  local name="$1"; shift
  if [ -f "runs/$name/manifest.json" ]; then
    echo "[skip] $name (manifest exists)"; return 0
  fi
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "${free_mib:-0}" -lt 9000 ]; then
    echo "[defer] $name (free=${free_mib}MiB < 9000MiB: linsam HVP would OOM; rerun queue after freeing)"
    return 0
  fi
  run "$name" "$@"
}

for S in 0 1 2; do
  echo "=== SEED $S unit ($(date '+%F %T')) ==="
  run "cifar100_vittiny_eos_sgd_s$S"       --opt sgd   --seed $S $COMMON
  run "cifar100_vittiny_eos_rankkx_d1_s$S" --opt rankkx --edit damp --k 1 --seed $S $COMMON
  run_gated "cifar100_vittiny_eos_linsam_s$S" --opt linsam --rho 0.05 --seed $S $COMMON
  run "cifar100_vittiny_eos_sam_s$S"       --opt sam   --rho 0.05 --seed $S $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
