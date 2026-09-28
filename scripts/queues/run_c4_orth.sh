#!/usr/bin/env bash
# C4 orthogonal-damping sweep: d ∈ {1, 1.5, 2, 4, 8} × seeds {0..4}.
# d=1 ⇒ vanilla SGD (sanity: λ_max pin at 2/η ≈ 40).
# d=8 ⇒ strong orthogonal damping (question: diverge or cap-robust?).
set -e
cd "$(dirname "$0")/../.."

DS=(1 1.5 2 4 8)
SEEDS=(0 1 2 3 4)
mkdir -p runs/c4

echo "========================================"
echo "C4 orth-damp sweep — $(date)"
echo "d ∈ ${DS[*]}  seeds ∈ ${SEEDS[*]}"
echo "========================================"

for D in "${DS[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo "[$(date +%H:%M:%S)] c4 d=$D seed=$S ..."
        D_VALUE=$D python -u scripts/experiments/plants_c4_lagged_experiment.py \
            --exp c4 --seed $S \
            > runs/c4/d${D}_s${S}.log 2>&1
        echo "[$(date +%H:%M:%S)] d=$D s=$S done"
    done
done
echo "========================================"
echo "ALL C4 DONE: $(date)"
echo "========================================"
