"""Run + analyze a multi-seed trajectory-alignment experiment (Gap E).

Step 1: run the EoS config under several seeds (same data, full-batch GD):
    python scripts/experiments/run_multiseed.py --seeds 0,1,2,3,4 --steps 4000
Step 2: fit a shared PCA and overlay / quantify alignment:
    python scripts/analysis/analyze_multiseed.py --sweep-dir runs/multiseed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.analysis.multiseed import (
    load_multiseed,
    overlay_figure,
    per_run_stats,
    shared_subspace,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="runs/multiseed")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    data, runs = load_multiseed(args.sweep_dir)
    if not data:
        print(f"no runs under {args.sweep_dir} — run scripts/experiments/run_multiseed.py first")
        return
    print(f"loaded {len(data)} runs: {[r.name for r in runs]}")

    list_W = [R["W"] for R in data]
    comps, mu, expl = shared_subspace(list_W, n_components=2)
    print(f"\nshared PCA (pooled over {len(data)} runs): "
          f"PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  PC1+2={expl[:2].sum()*100:.1f}%")

    out = Path(args.out)
    stats = overlay_figure(data, comps, mu, expl, out / "12_gapE_multiseed_alignment.png")
    stats["run"] = [r.name for r in runs]
    stats = stats[["run", "seed", "shared_pc1", "shared_pc12", "own_pc1", "own_pc12", "cos_pc1", "subspace_2d_cos"]]
    stats.to_csv(out / "gapE_subspace_stats.csv", index=False)

    print("\n=== per-run subspace alignment ===")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nmean shared-PC1 capture = {stats['shared_pc1'].mean():.3f}")
    print(f"mean shared-PC1+2 capture = {stats['shared_pc12'].mean():.3f}")
    print(f"mean cos(own-2D, shared-2D) = {stats['subspace_2d_cos'].mean():.3f}  (→1 = common skeleton)")


if __name__ == "__main__":
    main()
