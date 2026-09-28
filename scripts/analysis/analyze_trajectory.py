"""Run the Gap-A trajectory + sharpness co-analysis on a finished run.

Usage:
    .venv/bin/python scripts/analysis/analyze_trajectory.py --run mlp_ts_eos --color-by lambda_max
    .venv/bin/python scripts/analysis/analyze_trajectory.py --run mlp_ts_eos --color-by wnorm_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.analysis.trajectory import delta_vtop_alignment, load_run
from steos.analysis.visualize import make_alignment_figure, make_gap_a_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--color-by", default="lambda_max")
    ap.add_argument("--method", default="pca", choices=["pca", "umap"])
    args = ap.parse_args()

    run_dir = Path(args.run_root) / args.run
    R = load_run(run_dir)
    fig_dir = run_dir / "figures"

    expl = make_gap_a_figure(R, fig_dir / f"gapA_{args.method}_{args.color_by}.png",
                             color_by=args.color_by, method=args.method)
    print(f"PCA explained variance: PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  "
          f"PC1+PC2={expl[:2].sum()*100:.1f}%  (low value ⇒ trajectory is low-dim)")

    # also produce a ‖W_out‖-colored variant if not already
    if args.color_by != "wnorm_output":
        make_gap_a_figure(R, fig_dir / f"gapA_{args.method}_wnorm_output.png",
                          color_by="wnorm_output", method=args.method)
        print(f"saved gapA_{args.method}_wnorm_output.png")

    res = delta_vtop_alignment(R)
    if res is not None:
        vsteps, angles = res
        mask = ~np.isnan(angles)
        print(f"|cos(Δθ,v_top)|: init={np.nanmean(angles[:5]):.3f}  "
              f"final={np.nanmean(angles[-5:]):.3f}  "
              f"(Damian self-stabilization ⇒ →1 near EoS)")
        make_alignment_figure(R, fig_dir / "delta_vtop_alignment.png")
        print("saved delta_vtop_alignment.png")
    else:
        print("[note] no vtops.npz — re-run training with sharpness.enabled to get the alignment figure")


if __name__ == "__main__":
    main()
