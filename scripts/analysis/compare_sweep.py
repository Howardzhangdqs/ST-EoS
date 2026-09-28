"""Generate comparison figures for a finished sweep.

Usage:
    python scripts/analysis/compare_sweep.py --sweep-dir runs/sweep_optimizer_lr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.analysis.compare import (
    plot_lambda_max_across_lr,
    plot_pc_summary,
    plot_trajectories_grid,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    args = ap.parse_args()
    sd = Path(args.sweep_dir)
    fig = sd / "figures"
    plot_lambda_max_across_lr(sd, fig / "lambda_max_across_lr.png")
    plot_trajectories_grid(sd, fig / "trajectories_grid.png")
    plot_pc_summary(sd, fig / "pc_summary.png")
    print(f"comparison figures -> {fig}")


if __name__ == "__main__":
    main()
