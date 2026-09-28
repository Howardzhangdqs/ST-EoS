#!/usr/bin/env bash
# Foreground wait helper for the review-round-3 queue (batch sharpness +
# linearized SAM).  Usage: wait_r3.sh <target_manifests> [max_seconds]
# Exits 0 when >= target manifests exist (or QUEUE COMPLETE), 2 on anomaly
# (a run rc!=0 or tmux session eosr3 gone), 3 on local timeout (call again).
set -u
cd "$(dirname "$0")/../.."
TARGET=${1:?need target manifest count}
MAXS=${2:-560}
QLOG=runs/revision2/queue_r3_stdout.log
t0=$SECONDS
while true; do
  n=$(ls runs/cifar100_r18_sgd_detarpack_mb2_s*/manifest.json \
        runs/cifar100_r18_linsam_s*/manifest.json 2>/dev/null | wc -l)
  if grep -q "QUEUE COMPLETE" "$QLOG" 2>/dev/null; then
    echo "DONE: queue complete, manifests=$n"; exit 0
  fi
  if grep -q "rc=[^0]" "$QLOG" 2>/dev/null; then
    echo "ANOMALY: non-zero rc:"; grep "rc=" "$QLOG" | tail -3; exit 2
  fi
  if ! tmux has-session -t eosr3 2>/dev/null; then
    echo "ANOMALY: tmux session eosr3 gone"; tail -3 "$QLOG"; exit 2
  fi
  LOG=$(ls -t runs/revision2/log_cifar100_r18_sgd_detarpack_mb2_s*.txt \
              runs/revision2/log_cifar100_r18_linsam_s*.txt 2>/dev/null | head -1)
  last=$(tail -1 "$LOG" 2>/dev/null | awk '{print $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  echo "$(date '+%T') manifests=$n/$TARGET run=$(basename "${LOG:-?}" | sed 's/log_cifar100_r18_//;s/.txt//') step=${last:-?} gpu=${gpu}%"
  if [ "$n" -ge "$TARGET" ]; then echo "MILESTONE: $n manifests >= $TARGET"; exit 0
  fi
  if [ $((SECONDS - t0)) -ge "$MAXS" ]; then echo "TIMEOUT-LOCAL (call again)"; exit 3; fi
  sleep 90
done
