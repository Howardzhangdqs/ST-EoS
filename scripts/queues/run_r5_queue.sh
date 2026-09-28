#!/usr/bin/env bash
# ============================================================================
# Review-round-5 experiment queue (sequential, single RTX 5080).
#
# Stage A  batch sharpness ACROSS ARMS (reviewer priority 1, the
#          acceptance-critical ask): rerun the central stochastic arms with
#          --multibatch so the EoSS-regulated quantity E_B[g_BᵀH_Bg_B/|g_B|²]
#          (lhg_bs_mean/std, m=4) is measured on every arm, not just SGD
#          (Finding 43a). Arms: sam, linsam (n=5 — also closes the review's
#          n=3→5 ask for linSAM), amplify α=0.5 (rank-1 amplification),
#          rankkx damp k=1 (rank-1 damping). outcome-every 200 matches the
#          sgd_detarpack_mb2 reference exactly. sam/linsam arms additionally
#          log the NEW Taylor residual (see below) at ρ=0.05.
# Stage B  ρ-sweep, full SAM vs linearized SAM (reviewer priority 3):
#          ρ ∈ {0.005, 0.01, 0.02, 0.1} (0.05 exists) × {sam, linsam} × 3
#          seeds. The training script now logs the Taylor residual
#          ‖∇L_B(θ+ρĝ) − (g+ρH_Bĝ)‖/‖g‖ (taylor_res_rel) + its linearized
#          and total parts at every measurement point — the reviewer's
#          requested convergence/divergence diagnostic as ρ→0.
# Stage C  n=3→5 top-ups (reviewer priority 4): rankkx amplify k=5 / k=20
#          seeds 3-4 and current-batch-sensing damp-k1 seeds 3-4, under the
#          EXACT legacy names/flags so the existing analyses pick up n=5.
#
# Resume-safe: a run is skipped if its manifest.json already exists.
# Logs: runs/revision2/log_<run>.txt; queue stdout queue_r5_stdout.log.
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
echo "=== STAGE A1: SAM + multibatch (batch sharpness + Taylor residual, ρ=0.05) ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_sam_mb_s$S" --opt sam --rho 0.05 --seed $S --steps 5000 \
      --outcome-every 200 --multibatch $DET
done

echo "=== STAGE A2: linSAM + multibatch, n=5 (batch sharpness + Taylor residual) ($(date '+%F %T')) ==="
for S in 0 1 2 3 4; do
  run "cifar100_r18_linsam_mb_s$S" --opt linsam --rho 0.05 --seed $S --steps 5000 \
      --outcome-every 200 --multibatch $DET
done

echo "=== STAGE A3: rank-1 amplification α=0.5 + multibatch ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_amp05_mb_s$S" --opt amplify --amplify 0.5 --seed $S --steps 5000 \
      --outcome-every 200 --multibatch $DET
done

echo "=== STAGE A4: rank-1 damping k=1 + multibatch ($(date '+%F %T')) ==="
for S in 0 1 2; do
  run "cifar100_r18_rankkx_d1_mb_s$S" --opt rankkx --edit damp --k 1 \
      --seed $S --steps 5000 --outcome-every 200 --multibatch $DET
done

# ---------------------------------------------------------------------------
echo "=== STAGE B: ρ-sweep, full SAM vs linearized SAM (ρ=0.005/0.01/0.02/0.1; 0.05 exists) ($(date '+%F %T')) ==="
for R in 0.005 0.01 0.02 0.1; do
  for S in 0 1 2; do
    run "cifar100_r18_sam_rho${R}_s$S" --opt sam --rho $R --seed $S --steps 5000 \
        --outcome-every 100 $DET
  done
  for S in 0 1 2; do
    run "cifar100_r18_linsam_rho${R}_s$S" --opt linsam --rho $R --seed $S --steps 5000 \
        --outcome-every 100 $DET
  done
done

# ---------------------------------------------------------------------------
echo "=== STAGE C: n=3→5 top-ups (legacy names/flags) ($(date '+%F %T')) ==="
for S in 3 4; do
  run "cifar100_r18_rankkx_a5_s$S" --opt rankkx --edit amplify --amplify 1.0 \
      --k 5 --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 3 4; do
  run "cifar100_r18_rankkx_a20_s$S" --opt rankkx --edit amplify --amplify 1.0 \
      --k 20 --plant-N 40 --seed $S --steps 5000 --outcome-every 100 $DET
done
for S in 3 4; do
  run "cifar100_r18_rankkx_d1_curb_s$S" --opt rankkx --edit damp --k 1 \
      --sense-current --seed $S --steps 5000 --outcome-every 100 $DET
done

echo "=== QUEUE COMPLETE $(date '+%F %T') ==="