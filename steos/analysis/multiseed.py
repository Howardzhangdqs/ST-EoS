"""Multi-seed trajectory alignment (Gap E) — Song & Yun (NeurIPS 2023) test.

Question: do GD trajectories from different initializations share a common
low-dimensional "skeleton" (especially in the EoS phase)?

Method:
  * run the same config under several seeds (same data, same η, full-batch GD),
  * concatenate all snapshot matrices and fit ONE PCA → a shared subspace,
  * project every run onto the shared subspace and overlay (centered at each
    run's own start, so we compare shapes, not absolute coordinates),
  * quantify: (a) how much of each run's variance the SHARED top-k captures,
    (b) the principal angles between each run's OWN top-k subspace and the
    shared one. Small angles + high shared capture ⇒ a common skeleton.

Performance: PCA is done via the dual (N×N Gram-matrix eigendecomposition),
which is far cheaper than a full SVD when the number of snapshots N is much
smaller than the parameter count P (here N≈hundreds–thousands, P≈71 k).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .trajectory import load_run


def _pca_dual(Wc: np.ndarray):
    """PCA of a centered [N, P] matrix via the N×N Gram eigendecomposition.

    Returns (Vt[N,P] right singular vectors as rows (descending singular value),
             expl variance ratios descending).
    Works when N <= P; if N > P we transpose and translate back.
    """
    N, P = Wc.shape
    transpose = N > P
    M = Wc.T if transpose else Wc          # work on the smaller-dim side
    G = M @ M.T                             # [min, min]
    lam, U = np.linalg.eigh(G)             # ascending
    lam = np.clip(lam[::-1], 0.0, None)    # descending, non-negative
    U = U[:, ::-1]
    s = np.sqrt(lam)
    expl = lam / lam.sum() if lam.sum() > 0 else lam * 0.0
    safe = np.where(s > 1e-12, s, 1.0)
    V = (M.T @ U) / safe[None, :]          # right singular vectors as columns [P_or_N, min]
    if transpose:
        # singular vectors of Wc are U (already [P, P]); take first N as Vt
        Vt = U.T
    else:
        Vt = V.T                            # [min(N,P), P]
    return Vt, expl


def _displacements(list_W):
    """dW_k = W_k − W_k[0]: each trajectory expressed as a displacement from its
    own initialization. This removes the dominant between-run offset (different
    inits are far apart in parameter space) so the shared PCA captures the
    WITHIN-run dynamics — which is what trajectory alignment is about."""
    return [W - W[0:1] for W in list_W]


def shared_subspace(list_W, n_components: int = 2):
    list_dW = _displacements(list_W)
    Wall = np.concatenate(list_dW, axis=0)      # [ΣS, P], each starts at 0
    mu = Wall.mean(0)                            # small, but subtract for PCA correctness
    Wc = Wall - mu
    Vt, expl = _pca_dual(Wc)
    comps = Vt[:n_components]                     # [n, P], orthonormal rows
    return comps, mu, expl


def _subspace_min_cos(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine of the LARGEST principal angle between row-subspaces A, B
    (each rows-orthonormal). 1 ⇒ coincident planes, 0 ⇒ orthogonal."""
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    return float(s.min())


def per_run_stats(list_W, comps, mu) -> pd.DataFrame:
    k2 = comps.shape[0]
    list_dW = _displacements(list_W)
    rows = []
    for k, dW in enumerate(list_dW):
        Wc = dW - mu
        var_total = float((Wc ** 2).sum())
        proj = Wc @ comps.T                      # [S, n]
        var_shared = (proj ** 2).sum(axis=0)
        Vt_k, expl_k = _pca_dual(Wc)
        rows.append(dict(
            seed=k,
            shared_pc1=float(var_shared[0] / var_total),
            shared_pc12=float(var_shared[:k2].sum() / var_total),
            own_pc1=float(expl_k[0]),
            own_pc12=float(expl_k[:k2].sum()),
            cos_pc1=abs(float(Vt_k[0] @ comps[0])),
            subspace_2d_cos=_subspace_min_cos(Vt_k[:k2], comps[:k2]),
        ))
    return pd.DataFrame(rows)


def load_multiseed(sweep_dir) -> tuple[list[dict], list[Path]]:
    sd = Path(sweep_dir)
    runs = sorted(
        [d for d in sd.iterdir() if (d / "snapshots.npz").exists()],
        key=lambda d: float(d.name.rsplit("_", 1)[-1]),
    )
    data = [load_run(d) for d in runs]
    return data, runs


def overlay_figure(data, comps, mu, expl_global, out_path, center_at_start: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for k, R in enumerate(data):
        dW = R["W"] - R["W"][0:1]               # displacement from own init
        coords = (dW - mu) @ comps.T             # [S, 2] in the shared frame
        col = cmap(k % 10)
        ax.plot(coords[:, 0], coords[:, 1], "-", lw=1.0, color=col, alpha=0.85, label=f"seed {k}")
        ax.scatter([coords[0, 0]], [coords[0, 1]], marker="o", s=45, ec="k", fc=col, zorder=5)
        ax.scatter([coords[-1, 0]], [coords[-1, 1]], marker="*", s=150, color=col,
                   edgecolors="k", lw=0.4, zorder=5)
    ax.set_xlabel(f"shared PC1 ({expl_global[0]*100:.1f}% of pooled displacement var)")
    ax.set_ylabel(f"shared PC2 ({expl_global[1]*100:.1f}% of pooled displacement var)")
    ax.set_title("Multi-seed DISPLACEMENT trajectories in the shared 2-D frame\n(each W_k − W_k[0], overlayed)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)

    stats = per_run_stats([R["W"] for R in data], comps, mu)
    ax2 = axes[1]
    x = np.arange(len(stats))
    w = 0.35
    ax2.bar(x - w/2, stats["shared_pc1"], w, label="captured by shared PC1", color="tab:purple")
    ax2.bar(x + w/2, stats["shared_pc12"], w, label="captured by shared PC1+2", color="tab:cyan")
    ax2.bar(x, stats["subspace_2d_cos"], 0.09, label="cos(angle): own-2D vs shared-2D", color="tab:red")
    ax2.set_xticks(x); ax2.set_xticklabels([f"seed {k}" for k in stats["seed"]])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("fraction / cosine")
    ax2.set_title("Does each run live in the shared subspace?")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Gap E — multi-seed trajectory alignment (Song & Yun 2023 skeleton test)", y=1.02)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return stats
