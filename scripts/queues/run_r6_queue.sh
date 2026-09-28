#!/usr/bin/env bash
# ============================================================================
# Review-round-6 experiment queue — TRIMMED PASS 3 (user-approved 2026-09-06).
#
# Original design: B∈{2048,512,128,32} × μ∈{0,0.9} × 3 arms × 3 seeds on the
# canonical 5000-step protocol (see git history / REVISION2_STATUS for
# passes 1-2). Measured timings put the full grid at ~70 GPU-h; the approved
# trim keeps every SENSING run (answers P1/P3/P4: gradient authority, EoSS
# interpolation, reference-λ decoupling) and the actuator arms (d1/amp) only
# at the DECISIVE P2 points: 128/μ0 (epoch-matched μ=0 mid point),
# 512/μ0.9 (epoch-matched μ=0.9 mid point), and BOTH 2048 cells (the
# quasi-deterministic end where sign-control may revive).
# DROPPED: d1/amp at bs=32 (sign-control inert a fortiori — the edited
# subspace's share falls below the dead 2e-4 anchor) and at 512/μ0.
#
# Quantities per cell (the reviewer's phase figure):
#   - gradient-energy share of the edited top-1 subspace  (plant_per_step.jsonl
#     cos_g_v in d1; gdotv/gnorm in amp05)
#   - damping outcome    = late reference lambda (d1)  vs sgd cell reference
#   - amplification outcome = late reference lambda (amp05) vs sgd
#   - batch sharpness E_B[g_B^T H_B g_B / |g_B|^2]      (sgd sensing arm)
#   - reference lambda_max (ARPACK outcome, fixed eval batch 512)
# The (128, 0.9) anchor cell reuses the round-3/5 runs:
#   sgd sensing  -> cifar100_r18_sgd_detarpack_mb2_s{0,1,2}
#   rank-1 damp  -> cifar100_r18_rankkx_d1_mb_s{0,1,2}
#   rank-1 amp   -> cifar100_r18_amp05_mb_s{0,1,2}
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout appended to
# queue_r6_stdout.log.
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
# Remaining arms only (everything else already has a manifest from pass 2).
# Cell order: cheap 32 sensing seeds first (completes the sensing grid),
# then the two 2048 transition cells (the decisive P2 points).
for S in 1 2; do
  run "cifar100_r18_bs32_m09_sgd_s$S" --opt sgd --bs 32 --mom 0.9 --seed $S $COMMON
  run "cifar100_r18_bs32_m0_sgd_s$S"  --opt sgd --bs 32 --mom 0.0 --seed $S $COMMON
done
for CELL in "2048 0.9 m09" "2048 0.0 m0"; do
  set -- $CELL; B=$1; M=$2; MT=$3
  echo "=== CELL bs=$B mu=$M ($(date '+%F %T')) ==="
  for S in 1 2; do
    run "cifar100_r18_bs${B}_${MT}_sgd_s$S" --opt sgd --bs $B --mom $M --seed $S $COMMON
  done
  for S in 0 1 2; do
    run "cifar100_r18_bs${B}_${MT}_d1_s$S" --opt rankkx --edit damp --k 1 \
        --bs $B --mom $M --seed $S $COMMON
  done
  for S in 0 1 2; do
    run "cifar100_r18_bs${B}_${MT}_amp05_s$S" --opt amplify --amplify 0.5 \
        --bs $B --mom $M --seed $S $COMMON
  done
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
