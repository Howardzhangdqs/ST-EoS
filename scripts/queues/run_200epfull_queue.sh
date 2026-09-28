#!/usr/bin/env bash
# ============================================================================
# 200-EPOCH FULL queue (200epf): exact re-run of the 200ep external-validity
# recipe with the trainer's new end-of-training additions — final accuracy on
# the FULL CIFAR-100 10k test set (manifest key final_acc_full) and persisted
# final weights (final_model.pt, state_dict incl. BN buffers).
#
# Identical hyperparameters/order to run_200ep_queue.sh (seed-interleaved,
# n=5 seeds × 3 arms: sgd / rankkx_d1 / sam); ONLY run names (_200epf_),
# per-run logs and the aggregate log differ. Original _200ep_ runs are never
# touched. Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt, runs/revision2/queue_200epf_stdout.log
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Attempt-1 launch OOMed: this GPU is SHARED — external containers hold
# ~8.4 GiB steady, leaving ~7 GiB. chunk 1024 was sized for a FREE 16 GB GPU
# (see _ARP_DEFAULT_CHUNK). chunk-size is a pure HVP memory-batching knob
# (algebraically exact under chunking, docstring in hessian.py) and is NOT a
# scientific hyperparameter; 256 quarters the double-backward graph peak.
# Science knobs (lr/mom/bs/rho/plant_N/schedule/outcome_every/eval_batch/edit)
# stay verbatim vs run_200ep_queue.sh.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 256"

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
STEPS_EP=391          # 50000/128 ≈ 390.6
STEPS_200EP=78200     # 200 × 391
MEAS_EVERY=1955       # every 5 epochs → 40 exact-ARPACK checkpoints
SEEDS="0 1 2 3 4"

echo "=== 200-EPOCH-FULL QUEUE (n=5, 3 arms, full-10k eval + final weights; $(date '+%F %T')) ==="
for S in $SEEDS; do
  run "cifar100_r18_sgd_200epf_s$S" --opt sgd --seed $S --augment \
      --schedule cosine --lr-min 0.0 --steps $STEPS_200EP \
      --outcome-every $MEAS_EVERY $DET
  run "cifar100_r18_rankkx_d1_200epf_s$S" --opt rankkx --edit damp --k 1 \
      --plant-N 50 --seed $S --augment --schedule cosine --lr-min 0.0 \
      --steps $STEPS_200EP --outcome-every $MEAS_EVERY $DET
  run "cifar100_r18_sam_200epf_s$S" --opt sam --rho 0.05 --seed $S --augment \
      --schedule cosine --lr-min 0.0 --steps $STEPS_200EP \
      --outcome-every $MEAS_EVERY $DET
done

echo "=== 200-EPOCH-FULL QUEUE COMPLETE $(date '+%F %T') ==="
