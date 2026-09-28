#!/usr/bin/env bash
# Lagged-phase plant sweep: k ∈ {0, 1, 2, 3} × seeds {0..4}.
# k=0 ⇒ accurate power iter (sanity: sharp-escape to λ_max ~263).
# k=1 ⇒ free-lagged Δθ_{t−1} (sanity: pin at 2/η ≈ 40).
# k=2,3 ⇒ new observation points (pin or diverge?).
set -e
cd "$(dirname "$0")/../.."

KS=(0 1 2 3)
SEEDS=(0 1 2 3 4)
mkdir -p runs/lagged

echo "========================================"
echo "lagged plant sweep — $(date)"
echo "k ∈ ${KS[*]}  seeds ∈ ${SEEDS[*]}"
echo "========================================"

for K in "${KS[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo "[$(date +%H:%M:%S)] lagged k=$K seed=$S ..."
        K_VALUE=$K python -u scripts/experiments/plants_c4_lagged_experiment.py \
            --exp lagged --seed $S \
            > runs/lagged/k${K}_s${S}.log 2>&1
        echo "[$(date +%H:%M:%S)] k=$K s=$S done"
    done
done
echo "========================================"
echo "ALL LAGGED DONE: $(date)"
echo "========================================"
