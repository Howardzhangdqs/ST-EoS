#!/usr/bin/env bash
# Foreground wait helper for the review-round-11 queue.
# Usage: wait_r11.sh <target_manifests> [max_seconds]
#   target 10 = all 10 r11 runs done (sam_mb s3,s4 + linsamneg s0-4 +
#               sgd_detarpack_mb2c s0-2; the s[34]/mb2c globs exclude the
#               pre-existing s0-2 sam_mb and mb2 runs).
# Exits 0 when >= target manifests exist (or QUEUE COMPLETE), 2 on anomaly
# (a run rc!=0 in queue_r11_stdout.log or tmux session eosr11 gone),
# 3 on local timeout (call again with the same args).
set -u
cd "$(dirname "$0")/../.."
TARGET=${1:-10}
MAXS=${2:-540}
QLOG=runs/revision2/queue_r11_stdout.log
t0=$SECONDS
while true; do
  n=$(ls runs/cifar100_r18_sam_mb_s[34]/manifest.json \
        runs/cifar100_r18_linsamneg_mb_s[0-4]/manifest.json \
        runs/cifar100_r18_sgd_detarpack_mb2c_s[0-4]/manifest.json 2>/dev/null | wc -l)
  if grep -q "QUEUE COMPLETE" "$QLOG" 2>/dev/null; then
    echo "DONE: queue complete, manifests=$n"; exit 0
  fi
  bad=$(grep "rc=[^0]" "$QLOG" 2>/dev/null | head -3)
  if [ -n "$bad" ]; then
    echo "ANOMALY: non-zero rc:"; echo "$bad"; exit 2
  fi
  if ! tmux has-session -t eosr11 2>/dev/null; then
    echo "ANOMALY: tmux session eosr11 gone"; tail -3 "$QLOG"; exit 2
  fi
  LOG=$(ls -t runs/revision2/log_cifar100_r18_sam_mb_s[34].txt \
        runs/revision2/log_cifar100_r18_linsamneg_mb_s[0-4].txt \
        runs/revision2/log_cifar100_r18_sgd_detarpack_mb2c_s[0-4].txt 2>/dev/null | head -1)
  last=$(tail -1 "$LOG" 2>/dev/null | awk '{print $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  echo "$(date '+%T') manifests=$n/$TARGET run=$(basename "${LOG:-?}" | sed 's/log_cifar100_r18_//;s/.txt//') step=${last:-?} gpu=${gpu}%"
  if [ "$n" -ge "$TARGET" ]; then echo "MILESTONE: $n manifests >= $TARGET"; exit 0; fi
  if [ $((SECONDS - t0)) -ge "$MAXS" ]; then echo "TIMEOUT-LOCAL (call again)"; exit 3; fi
  sleep 90
done
