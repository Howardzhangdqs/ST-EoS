#!/usr/bin/env bash
# ============================================================================
# Review-round-2 experiment queue (sequential, single RTX 5080).
#
# Stage 1  cross-batch confound study   : sgd --multibatch (seeds 0-2)
# Stage 2  current-batch sensing control: rankkx damp k=1 --sense-current (0-2)
# Stage 3  rank interpolation, damping  : rankkx damp k=1(+spec10) / 2 / 5 / 10 / 20
#          + sgd spectrum-k=10 baseline (redistribution diagnostic)
# Stage 4  rank interpolation, amplify  : rankkx amplify α=1 k=5 / k=20
# Stage 5  standard long recipe         : 100 epochs, cosine, augment, 3 arms
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOGDIR=runs/revision2
mkdir -p "$LOGDIR"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

DET="--deterministic --outcome-arpack --outcome-tol 1e-6 --outcome-k 3 \
     --eval-batch 512 --chunk-size 1024"

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
echo "=== STAGE 1: cross-batch sensing study ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_sgd_detarpack_mb_s$S" --opt sgd --seed $S --steps 5000 \
      --outcome-every 200 --multibatch $DET
done

# ---------------------------------------------------------------------------
echo "=== STAGE 2: current-batch-sensing rank-1 damping control ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d1_curb_s$S" --opt rankkx --edit damp --k 1 \
      --sense-current --seed $S --steps 5000 --outcome-every 100 $DET
done

# ---------------------------------------------------------------------------
echo "=== STAGE 3: rank interpolation, damping ($(date '+%F %T')) ==="
# k=1 with the top-10 spectrum + controller-overlap instrumentation (the
# spectral-redistribution diagnostic pairs with the sgd spectrum baseline)
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d1_s$S" --opt rankkx --edit damp --k 1 \
      --spectrum-k 10 --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 0 1 2; do
  run "cifar100_r18_sgd_detarpack_spec_s$S" --opt sgd --spectrum-k 10 \
      --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d2_s$S" --opt rankkx --edit damp --k 2 \
      --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d5_s$S" --opt rankkx --edit damp --k 5 \
      --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d10_s$S" --opt rankkx --edit damp --k 10 \
      --seed $S --steps 5000 --outcome-every 100 $DET
done

# ---------------------------------------------------------------------------
echo "=== STAGE 4: rank interpolation, amplification ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_rankkx_a5_s$S" --opt rankkx --edit amplify --amplify 1.0 \
      --k 5 --seed $S --steps 5000 --outcome-every 100 $DET
done
# k=20 arms use refresh N=40 (staleness scan N∈{10..100} invariant; halves cost)
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d20_s$S" --opt rankkx --edit damp --k 20 \
      --plant-N 40 --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 0 1 2; do
  run "cifar100_r18_rankkx_a20_s$S" --opt rankkx --edit amplify --amplify 1.0 \
      --k 20 --plant-N 40 --seed $S --steps 5000 --outcome-every 100 $DET
done

# ---------------------------------------------------------------------------
echo "=== STAGE 5: standard long recipe — 100 epochs, cosine, augment ($(date '+%F %T')) ==="
STEPS_EP=391          # 50000/128 ≈ 390.6
STEPS_100EP=39100
MEAS_EVERY=1955       # every 5 epochs
for S in 0 1 2; do
  run "cifar100_r18_sgd_100ep_s$S" --opt sgd --seed $S --augment \
      --schedule cosine --lr-min 0.0 --steps $STEPS_100EP \
      --outcome-every $MEAS_EVERY $DET
done
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d1_100ep_s$S" --opt rankkx --edit damp --k 1 \
      --plant-N 50 --seed $S --augment --schedule cosine --lr-min 0.0 \
      --steps $STEPS_100EP --outcome-every $MEAS_EVERY $DET
done
for S in 0 1 2; do
  run "cifar100_r18_sam_100ep_s$S" --opt sam --rho 0.05 --seed $S --augment \
      --schedule cosine --lr-min 0.0 --steps $STEPS_100EP \
      --outcome-every $MEAS_EVERY $DET
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
