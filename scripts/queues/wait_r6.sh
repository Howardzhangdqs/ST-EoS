#!/usr/bin/env bash
# Foreground wait helper for the review-round-6 regime-transition queue.
# Usage: wait_r6.sh <target_manifests> [max_seconds]
#   target 4    = pilot corners done
#   target 67   = all new runs done (63 grid + 4 pilot, pilot counted in cells)
# Exits 0 when >= target manifests exist (or QUEUE COMPLETE), 2 on anomaly
# (a run rc!=0 or tmux session eosr6 gone), 3 on local timeout (call again).
set -u
cd "$(dirname "$0")/../.."
TARGET=${1:?need target manifest count}
MAXS=${2:-560}
QLOG=runs/revision2/queue_r6_stdout.log
t0=$SECONDS
while true; do
  n=$(ls runs/cifar100_r18_bs*_sgd_s*/manifest.json \
        runs/cifar100_r18_bs*_d1_s*/manifest.json \
        runs/cifar100_r18_bs*_amp05_s*/manifest.json 2>/dev/null | wc -l)
  if grep -q "QUEUE COMPLETE" "$QLOG" 2>/dev/null; then
    echo "DONE: queue complete, manifests=$n"; exit 0
  fi
  # Anomaly = rc!=0 line NOT among the known non-scientific failures:
  # - the two bs=2048 pilot OOMs (create_graph bug, fixed 2026-09-05; re-run
  #   under the fixed code in pass 2)
  # - bs2048_m09_d1_s2 cudaErrorLaunchTimeout (transient GPU fault class,
  #   2026-09-06 23:12; sibling seeds passed immediately before; no manifest
  #   → re-run in the pass-4 queue invocation)
  bad=$(grep "rc=[^0]" "$QLOG" 2>/dev/null \
        | grep -v "cifar100_r18_bs2048_m09_sgd_s0" \
        | grep -v "cifar100_r18_bs2048_m0_sgd_s0" \
        | grep -v "cifar100_r18_bs2048_m09_d1_s2" | head -3)
  if [ -n "$bad" ]; then
    echo "ANOMALY: non-zero rc:"; echo "$bad"; exit 2
  fi
  if ! tmux has-session -t eosr6 2>/dev/null; then
    echo "ANOMALY: tmux session eosr6 gone"; tail -3 "$QLOG"; exit 2
  fi
  LOG=$(ls -t runs/revision2/log_cifar100_r18_bs*_s*.txt 2>/dev/null | head -1)
  last=$(tail -1 "$LOG" 2>/dev/null | awk '{print $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  echo "$(date '+%T') manifests=$n/$TARGET run=$(basename "${LOG:-?}" | sed 's/log_cifar100_r18_//;s/.txt//') step=${last:-?} gpu=${gpu}%"
  if [ "$n" -ge "$TARGET" ]; then echo "MILESTONE: $n manifests >= $TARGET"; exit 0; fi
  if [ $((SECONDS - t0)) -ge "$MAXS" ]; then echo "TIMEOUT-LOCAL (call again)"; exit 3; fi
  sleep 90
done
