#!/usr/bin/env bash
# ============================================================================
# Chain watcher r10c -> r10d (tmux infrastructure, zero API usage).
# Precedent: chain789 watcher (REVISION2_STATUS, rounds 7-9).
#
# Flow:
#   1. Wait for the r10c queue to print QUEUE COMPLETE
#      (abort if a NON-linsam run reports rc!=0, or the eosr10c tmux session
#      dies without COMPLETE — linsam rc=1/[defer] are known-handled).
#   2. GPU smoke of the chunked-linsam path (250 steps @ bs512, chunk 128).
#      Abort if it fails (keep the run dir + log for debugging).
#   3. Launch the r10d linsam queue (3 seeds, memory-gated, resume-safe).
#
# Log: runs/revision2/chain_r10cd_stdout.log (tmux session eosr10chain).
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
QLOG=runs/revision2/queue_r10c_stdout.log

echo "[chain] watching r10c $(date '+%F %T')"
while true; do
  if grep -q "QUEUE COMPLETE" "$QLOG" 2>/dev/null; then break; fi
  if grep "rc=[^0]" "$QLOG" 2>/dev/null | grep -q -v "_linsam_"; then
    echo "[chain] ABORT: non-linsam rc!=0 in r10c:"; grep "rc=" "$QLOG" | tail -3; exit 2
  fi
  if ! tmux has-session -t eosr10c 2>/dev/null; then
    echo "[chain] ABORT: eosr10c gone without COMPLETE"; tail -3 "$QLOG"; exit 2
  fi
  sleep 120
done
echo "[chain] r10c COMPLETE $(date '+%F %T') — chunked-linsam GPU smoke"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
.venv/bin/python scripts/experiments/train_cifar100_resnet18.py \
  --run-name smoke_linsam_chunked_b512 \
  --opt linsam --rho 0.05 --linsam-chunk 128 --seed 9 --lr 0.05 --bs 512 \
  --model vit_tiny --steps 250 --outcome-every 200 --deterministic \
  --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 --eval-batch 512 \
  --chunk-size 128 \
  > runs/revision2/log_smoke_linsam_chunked_b512.txt 2>&1
rc=$?
echo "[chain] smoke rc=$rc  $(date '+%F %T')"
if [ $rc -ne 0 ]; then
  echo "[chain] ABORT: chunked-linsam smoke failed (log_smoke_linsam_chunked_b512.txt kept)"
  exit 3
fi
rm -rf runs/smoke_linsam_chunked_b512

echo "[chain] launching r10d $(date '+%F %T')"
bash scripts/queues/run_r10d_linsam_chunked.sh
echo "[chain] r10d done rc=$?  $(date '+%F %T')"
