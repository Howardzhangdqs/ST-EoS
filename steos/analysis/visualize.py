"""Joint sharpness × trajectory visualization (Gap A — the thesis figure).

Layout
------
Left  : 2-D PCA projection of the weight trajectory, points colored by the
        chosen per-step quantity (λ_max / λ_c / ‖W_out‖ / ‖Δθ‖). Start ● and
        end ★ marked. PC explained variance on the axes answers "is the
        trajectory effectively low-dimensional?".
Right : three stacked time-series sharing the step axis —
        (1) loss (log)                — long-run decrease, short-scale wobble
        (2) λ_max & λ_c + 2/η line    — EoS signature + alignment convergence
        (3) ‖W_out‖ & ‖Δθ‖            — proxy sharpness vs update magnitude
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .trajectory import color_at_snapshots, pc_speed, project


def make_gap_a_figure(run_data: dict, out_path, color_by: str = "lambda_max", method: str = "pca"):
    coords, _, expl, _ = project(run_data, method=method, n_components=2)
    steps = run_data["snap_steps"]
    m = run_data["metrics"]
    lr = run_data["lr"]
    two_over_eta = 2.0 / lr

    try:
        c = color_at_snapshots(run_data, color_by)
        norm = mcolors.LogNorm(vmin=max(c.min(), 1e-6), vmax=c.max()) if _is_pos(c) else None
    except ValueError:
        c = steps.astype(float)
        norm = None

    fig = plt.figure(figsize=(13, 5.6))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1], hspace=0.32, wspace=0.22)
    axT = fig.add_subplot(gs[:, 0])

    sc = axT.scatter(coords[:, 0], coords[:, 1], c=c, cmap="plasma", s=16, norm=norm, zorder=3)
    axT.plot(coords[:, 0], coords[:, 1], color="0.75", lw=0.6, zorder=1)
    axT.scatter([coords[0, 0]], [coords[0, 1]], marker="o", s=90, ec="black", fc="white",
                lw=1.2, zorder=5, label="start")
    axT.scatter([coords[-1, 0]], [coords[-1, 1]], marker="*", s=240, c="red",
                edgecolors="black", lw=0.5, zorder=5, label="end")
    axT.set_xlabel(f"PC1 ({expl[0] * 100:.1f}% var)")
    axT.set_ylabel(f"PC2 ({expl[1] * 100:.1f}% var)")
    axT.set_title(f"Weight trajectory in 2-D {method.upper()}  —  color = {color_by}")
    axT.legend(fontsize=8, loc="best")
    cb = fig.colorbar(sc, ax=axT, fraction=0.046, pad=0.04)
    cb.set_label(color_by)
    axT.grid(alpha=0.25)

    ax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 1], sharex=ax0)

    ax0.plot(m["step"], m["loss"], lw=0.7, color="tab:blue")
    ax0.set_ylabel("loss")
    ax0.set_yscale("log")
    ax0.grid(alpha=0.25)

    sh = run_data["sharp"]
    if sh is not None:
        if "lambda_max" in sh:
            ax1.plot(sh["step"], sh["lambda_max"], "o-", ms=3, lw=0.7, color="tab:purple", label="λ_max")
        if "lambda_c" in sh:
            ax1.plot(sh["step"], sh["lambda_c"], "s-", ms=3, lw=0.7, color="tab:cyan", label="λ_c")
        ax1.axhline(two_over_eta, color="red", ls="--", lw=1.1, label=f"2/η = {two_over_eta:g}")
        ax1.legend(fontsize=7, loc="best")
    ax1.set_ylabel("sharpness")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.25)

    ax2.plot(m["step"], m["wnorm_output"], lw=0.7, color="tab:green", label="‖W_out‖  (proxy)")
    ax2.plot(m["step"], m["update_norm"], lw=0.6, color="tab:orange", alpha=0.75, label="‖Δθ‖")
    ax2.set_ylabel("norm")
    ax2.set_xlabel("step")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(alpha=0.25)
    ax2.set_xlim(m["step"].min(), m["step"].max())

    fig.suptitle(f"Gap-A: sharpness × weight-trajectory  —  run '{run_data['run_dir'].name}'", y=1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return expl


def make_alignment_figure(run_data: dict, out_path):
    """Bonus: |cos(Δθ, v_top)| over training (Damian self-stabilization test)."""
    from .trajectory import delta_vtop_alignment
    res = delta_vtop_alignment(run_data)
    if res is None:
        return None
    vsteps, angles = res
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(vsteps, angles, "o-", ms=4, lw=0.8, color="tab:red")
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|\cos(\Delta\theta,\, v_{\mathrm{top}})|$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Update–top-eigenvector alignment  (→1 at EoS = Damian self-stabilization)")
    ax.grid(alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return vsteps, angles


def _is_pos(c: np.ndarray) -> bool:
    return bool(np.all(c > 0))
