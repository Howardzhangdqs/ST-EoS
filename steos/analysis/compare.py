"""Cross-run comparison figures for sweeps (Gap-A corridor-shape story).

  * `plot_lambda_max_across_lr` — λ_max(t) for every LR with each 2/η threshold.
    Shows EoS threshold-tracking across learning rates and, at high η, the
    period-doubling / route-to-chaos signature (Kalra et al. ICLR 2025).
  * `plot_trajectories_grid`    — small multiples of the PC1/PC2 trajectory per
    variant, colored by step. Shows how the corridor geometry changes with η.
  * `plot_pc_summary`           — PC1/PC2 explained variance and alignment vs LR.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .trajectory import pca_project, load_run


def _variants(sweep_dir):
    sweep_dir = Path(sweep_dir)
    return sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and (d / "snapshots.npz").exists()],
        key=lambda d: float(d.name.rsplit("_", 1)[-1]),
    )


def plot_lambda_max_across_lr(sweep_dir, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    variants = _variants(sweep_dir)
    n = len(variants)
    for i, d in enumerate(variants):
        R = load_run(d)
        if R["sharp"] is None or "lambda_max" not in R["sharp"]:
            continue
        sh = R["sharp"].dropna(subset=["lambda_max"])
        lr = R["lr"]
        color = cmap(i / max(n - 1, 1))
        ax.plot(sh["step"], sh["lambda_max"], "-", lw=1.0, color=color,
                label=f"η={lr:g}  (2/η={2/lr:g})")
        ax.axhline(2 / lr, color=color, ls=":", lw=0.8, alpha=0.6)
    # annotate diverged runs
    for d in _variants(sweep_dir):
        R = load_run(d)
        if np.isfinite(R["W"]).all():
            continue
        ax.text(0.98, 0.02 + 0.05 * _variants(sweep_dir).index(d) * 0,
                f"η={R['lr']:g}: diverged", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="0.5")
    ax.set_xlabel("step")
    ax.set_ylabel("λ_max  (Hessian top eigenvalue)")
    ax.set_yscale("log")
    ax.set_title("λ_max(t) across learning rates — EoS threshold-tracking")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_trajectories_grid(sweep_dir, out_path):
    variants = [d for d in _variants(sweep_dir)
                if np.isfinite(load_run(d)["W"]).all()]  # skip diverged (NaN) runs
    n = len(variants)
    if n == 0:
        return
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for idx, d in enumerate(variants):
        ax = axes[idx // cols][idx % cols]
        R = load_run(d)
        coords, _, expl, _ = pca_project(R["W"], 2)
        st = R["snap_steps"]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=st, cmap="plasma", s=10)
        ax.plot(coords[:, 0], coords[:, 1], color="0.8", lw=0.4, zorder=0)
        ax.scatter([coords[0, 0]], [coords[0, 1]], marker="o", s=50, ec="k", fc="white", zorder=5)
        ax.scatter([coords[-1, 0]], [coords[-1, 1]], marker="*", s=140, c="red", zorder=5)
        ax.set_title(f"η={R['lr']:g}  (PC1={expl[0]*100:.0f}%, PC2={expl[1]*100:.0f}%)", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    for j in range(len(variants), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Weight trajectory (2-D PCA) across learning rates", y=1.01)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_pc_summary(sweep_dir, out_path):
    summary = pd.read_csv(Path(sweep_dir) / "summary.csv").sort_values("lr")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(summary["lr"], summary["pc1"], "o-", label="PC1")
    axes[0].plot(summary["lr"], summary["pc2"], "s-", label="PC2")
    axes[0].set_xscale("log"); axes[0].set_xlabel("learning rate η")
    axes[0].set_ylabel("explained variance"); axes[0].set_title("trajectory dimensionality")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(summary["lr"], summary["eos_frac"], "o-", color="tab:purple")
    axes[1].set_xscale("log"); axes[1].set_xlabel("η"); axes[1].set_ylabel("frac steps at 2/η (±25%)")
    axes[1].set_title("EoS occupancy"); axes[1].grid(alpha=0.3)

    axes[2].plot(summary["lr"], summary["align_final"], "o-", color="tab:red")
    axes[2].set_xscale("log"); axes[2].set_xlabel("η")
    axes[2].set_ylabel(r"$|\cos(\Delta\theta, v_{top})|$ (final)")
    axes[2].set_title("update–v_top alignment at end"); axes[2].grid(alpha=0.3)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
