#!/usr/bin/env bash
# ============================================================================
# Review-round-10d — linsam arm of the ViT EoS phase-2, chunked HVP path.
#
# Completes the r10c 4-arm design: linsam at bs512 OOMs on the legacy
# single-graph double-backward (needs ~11 GiB; GPU shared with ~4.5 GiB of
# cross-namespace external processes, r10c smoke 2026-09-14 19:30 rc=1, and
# the r10c in-queue attempt rc=1 at 20:37). The chunked path
# (linsam_step_chunked, --linsam-chunk 128) restricts the double-backward
# graph to 128 samples at a time — peak memory ≈ a bs128 linsam step.
# Math verified against the legacy path on a CPU unit test (15 steps,
# 2 coupling configs, max relative deviation 2.3e-7 — float associativity
# only; /tmp/opencode/test_chunked_linsam.py, 2026-09-14 21:10).
#
# Runs: cifar100_vittiny_eos_linsam_s{0,1,2} — the SAME run names as the
# r10c linsam slots (no manifest exists for them: s0 OOM-failed, s1/s2
# deferred), so r10c+r10d together form the complete 4-arm × 3-seed grid.
# Protocol identical to r10c: bs512, lr 0.05 (2/eta=40), 5000 steps,
# deterministic ARPACK k=3 tol 1e-6, eval batch 512, chunk 128 (eval),
# outcome every 200. LayerNorm model → the chunked path's 1+C train-mode
# forwards per step have no running-stat side effects.
#
# Gate: >= 6500 MiB free (chunked peak ~6.5-7 GiB).
# Expected cost: ~55 min/run → ~2.75 h for 3 runs.
# Resume-safe: skip a run if its manifest.json exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r10d_stdout.log.
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

run_gated () {
  local name="$1"; shift
  if [ -f "runs/$name/manifest.json" ]; then
    echo "[skip] $name (manifest exists)"; return 0
  fi
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "${free_mib:-0}" -lt 6500 ]; then
    echo "[defer] $name (free=${free_mib}MiB < 6500MiB; rerun queue when GPU is freer)"
    return 0
  fi
  local t0=$(date +%s)
  echo "[start] $name  $(date '+%F %T')"
  "$PY" scripts/experiments/train_cifar100_resnet18.py --run-name "$name" "$@" \
      > "$LOGDIR/log_$name.txt" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "[done ] $name  rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%F %T')"
}

for S in 0 1 2; do
  run_gated "cifar100_vittiny_eos_linsam_s$S" \
      --opt linsam --rho 0.05 --linsam-chunk 128 --seed $S $COMMON
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
