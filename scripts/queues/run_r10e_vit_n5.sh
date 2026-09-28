#!/usr/bin/env bash
# ============================================================================
# Review-round-10e — ViT-tiny EoS-band n=5 top-up: seeds 3 & 4.
# User-authorized supplementary runs (2026-09-20); paper/ and all existing
# runs/ dirs untouched — this queue only creates new run dirs.
#
# EXACT replication of the r10c EoS-band protocol (see
# scripts/queues/run_r10c_vit_eos_phase2.sh and runs/cifar100_vittiny_eos_*_s{0,1,2}
# manifests); only the seed changes. Operating point bs=512, lr=0.05
# (2/eta = 40), 5000 steps, deterministic CUDA + ARPACK (k=3, tol 1e-6),
# eval batch 512, chunk 128, constant lr, outcome every 200, no augment.
# Arms: sgd / rankkx_d1 (--edit damp --k 1; plant_N=20 default,
# lam_targ auto = 2/lr = 40) / linsam (rho=0.05) / sam (rho=0.05).
#
# Seed-interleaved (seed 3 four arms -> seed 4 four arms). Resume-safe:
# a run is [skip]ped iff runs/<name>/manifest.json exists. `set -u`, no
# `set -e`: one failed run never blocks the rest; rerun the queue to fill.
#
# MEMORY GATE for linsam (fixed 2026-09-20 backfill): s0-2 actually ran via
# the r10d CHUNKED path (--linsam-chunk 128, linsam_step_chunked; legacy
# unchunked double-backward needs ~13 GiB and OOMs at bs512 on the shared
# GPU — r10c in-queue attempts rc=1, r10e attempts 16:37/19:58 rc=1).
# Chunked equivalence vs legacy verified on CPU (max rel dev 2.3e-7,
# /tmp/opencode/test_chunked_linsam.py, 2026-09-14). Gate: >= 6500 MiB free
# (chunked peak ~6.5-7 GiB); otherwise log [defer] and continue — rerunning
# this queue after freeing picks it up automatically.
#
# Expected cost (r10c seeds 0-2 measured): sgd 27-32 min, sam 52-57 min,
# linsam 91-93 min, rankkx_d1 99-115 min → 2 seeds ≈ 9-10 h total.
# Logs: per run runs/<name>/log_run.txt; aggregate queue stdout with
# [start]/[done] rc=N lines at runs/revision2/queue_r10e_stdout.log.
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
mkdir -p runs/revision2
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # OOM-collision hardening (2026-09-20 backfill)

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 128"
COMMON="--model vit_tiny --lr 0.05 --bs 512 --steps 5000 --outcome-every 200 $DET"

run () {
  local name="$1"; shift
  if [ -f "runs/$name/manifest.json" ]; then
    echo "[skip] $name (manifest exists)"; return 0
  fi
  mkdir -p "runs/$name"
  local t0=$(date +%s)
  echo "[start] $name  $(date '+%F %T')"
  "$PY" scripts/experiments/train_cifar100_resnet18.py --run-name "$name" "$@" \
      > "runs/$name/log_run.txt" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "[done ] $name  rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%F %T')"
}

run_gated () {  # linsam: needs >= 9000 MiB free (r10c phantom-memory gate)
  local name="$1"; shift
  if [ -f "runs/$name/manifest.json" ]; then
    echo "[skip] $name (manifest exists)"; return 0
  fi
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "${free_mib:-0}" -lt 6500 ]; then
    echo "[defer] $name (free=${free_mib}MiB < 6500MiB: chunked linsam peak ~6.5-7 GiB; rerun queue after freeing)"
    return 0
  fi
  run "$name" "$@"
}

for S in 3 4; do
  echo "=== SEED $S unit ($(date '+%F %T')) ==="
  run "cifar100_vittiny_eos_sgd_s$S"       --opt sgd   --seed $S $COMMON
  run "cifar100_vittiny_eos_rankkx_d1_s$S" --opt rankkx --edit damp --k 1 --seed $S $COMMON
  run_gated "cifar100_vittiny_eos_linsam_s$S" --opt linsam --rho 0.05 --linsam-chunk 128 --seed $S $COMMON
  run "cifar100_vittiny_eos_sam_s$S"       --opt sam   --rho 0.05 --seed $S $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
