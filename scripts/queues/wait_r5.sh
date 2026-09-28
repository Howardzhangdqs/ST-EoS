#!/usr/bin/env bash
# Foreground wait helper for the review-round-5 queue (batch sharpness
# across arms + ρ-sweep + n=5 top-ups).  Usage:
#   wait_r5.sh <target_manifests> [max_seconds]
# Exit codes: 0 = queue COMPLETE or manifests ≥ target; 2 = anomaly (a run
# rc≠0 or tmux session eosr5 gone); 3 = local timeout (call again same args).
set -u
cd "$(dirname "$0")/../.."
TARGET=${1:?need target manifest count}
MAXS=${2:-560}
QLOG=runs/revision2/queue_r5_stdout.log
GLOBS="runs/cifar100_r18_sam_mb_s*/manifest.json \
       runs/cifar100_r18_linsam_mb_s*/manifest.json \
       runs/cifar100_r18_amp05_mb_s*/manifest.json \
       runs/cifar100_r18_rankkx_d1_mb_s*/manifest.json \
       runs/cifar100_r18_sam_rho*_s*/manifest.json \
       runs/cifar100_r18_linsam_rho*_s*/manifest.json \
       runs/cifar100_r18_rankkx_a5_s*/manifest.json \
       runs/cifar100_r18_rankkx_a20_s*/manifest.json \
       runs/cifar100_r18_rankkx_d1_curb_s*/manifest.json"
t0=$SECONDS
while true; do
  n=$(ls $GLOBS 2>/dev/null | wc -l)
  if grep -q "QUEUE COMPLETE" "$QLOG" 2>/dev/null; then
    echo "DONE: queue complete, manifests=$n"; exit 0
  fi
  if grep -q "rc=[^0]" "$QLOG" 2>/dev/null; then
    echo "ANOMALY: non-zero rc:"; grep "rc=" "$QLOG" | tail -3; exit 2
  fi
  if ! tmux has-session -t eosr5 2>/dev/null; then
    echo "ANOMALY: tmux session eosr5 gone"; tail -3 "$QLOG"; exit 2
  fi
  LOG=$(ls -t runs/revision2/log_cifar100_r18_sam_mb_s*.txt \
              runs/revision2/log_cifar100_r18_linsam_mb_s*.txt \
              runs/revision2/log_cifar100_r18_amp05_mb_s*.txt \
              runs/revision2/log_cifar100_r18_rankkx_d1_mb_s*.txt \
              runs/revision2/log_cifar100_r18_sam_rho*_s*.txt \
              runs/revision2/log_cifar100_r18_linsam_rho*_s*.txt \
              runs/revision2/log_cifar100_r18_rankkx_a5_s*.txt \
              runs/revision2/log_cifar100_r18_rankkx_a20_s*.txt \
              runs/revision2/log_cifar100_r18_rankkx_d1_curb_s*.txt \
              2>/dev/null | head -1)
  last=$(tail -1 "$LOG" 2>/dev/null | awk '{print $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  echo "$(date '+%T') manifests=$n/$TARGET run=$(basename "${LOG:-?}" | sed 's/log_cifar100_r18_//;s/.txt//') step=${last:-?} gpu=${gpu}%"
  if [ "$n" -ge "$TARGET" ]; then echo "MILESTONE: $n manifests >= $TARGET"; exit 0
  fi
  if [ $((SECONDS - t0)) -ge $MAXS ]; then echo "TIMEOUT-LOCAL (call again)"; exit 3; fi
  sleep 90
done
