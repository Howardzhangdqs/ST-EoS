"""Trajectory geometry analysis (Gap A core).

Loads the snapshot matrix W ∈ ℝ^{S×P} (S snapshots, P params) plus the
per-step metrics and the curvature time-series, and produces:
  * a 2-D PCA (or UMAP) projection of the weight trajectory,
  * curvature / weight-norm values interpolated onto the snapshot steps
    (so the trajectory can be colored by sharpness),
  * the per-snapshot speed ‖Δθ‖ in PC space,
  * (bonus) |cos(Δθ, v_top)| along training — the Damian et al. (2023)
    self-stabilization signature: at EoS the update should align with the
    top Hessian eigenvector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    snap = np.load(run_dir / "snapshots.npz", allow_pickle=True)
    W = snap["W"].astype(np.float64)            # [S, P], already ref-centered
    snap_steps = snap["steps"].astype(int)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    sharp = pd.read_csv(run_dir / "sharpness.csv") if (run_dir / "sharpness.csv").exists() else None
    vtops = np.load(run_dir / "vtops.npz", allow_pickle=True) if (run_dir / "vtops.npz").exists() else None
    dthetas = np.load(run_dir / "dthetas.npz", allow_pickle=True) if (run_dir / "dthetas.npz").exists() else None
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # lr is stored under config.optimizer.lr (MLP runs) or top-level (timm runs)
    if "lr" in manifest:
        lr = float(manifest["lr"])
    else:
        lr = float(manifest.get("config", {}).get("optimizer", {}).get("lr", 0.05))
    return dict(W=W, snap_steps=snap_steps, metrics=metrics, sharp=sharp, vtops=vtops,
                dthetas=dthetas, manifest=manifest, lr=lr, run_dir=run_dir)


def pca_project(W: np.ndarray, n_components: int = 2):
    """Economy-SVD PCA. Returns (coords[S,n], comps[n,P], explained_var_ratio, mean)."""
    mu = W.mean(0)
    Wc = W - mu
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    comps = Vt[:n_components]
    coords = Wc @ comps.T
    var = S ** 2
    expl = var / var.sum()
    return coords, comps, expl, mu


def umap_project(W: np.ndarray, n_components: int = 2, seed: int = 0):
    try:
        import umap  # umap-learn
    except ImportError as e:
        raise ImportError("umap-learn not installed; use pca_project or pip install umap-learn") from e
    reducer = umap.UMAP(n_components=n_components, random_state=seed, init="random")
    return reducer.fit_transform(W), None, np.array([np.nan] * n_components), W.mean(0)


def project(run_data: dict, method: str = "pca", n_components: int = 2):
    if method == "umap":
        return umap_project(run_data["W"], n_components)
    return pca_project(run_data["W"], n_components)


def interp_sharpness(sharp: pd.DataFrame, target_steps: np.ndarray, col: str) -> np.ndarray:
    s = sharp.dropna(subset=[col]).sort_values("step")
    return np.interp(target_steps, s["step"].to_numpy(float), s[col].to_numpy(float))


def color_at_snapshots(run_data: dict, color_by: str) -> np.ndarray:
    steps = run_data["snap_steps"]
    if color_by in ("lambda_max", "lambda_c", "eta_c"):
        if run_data["sharp"] is None or color_by not in run_data["sharp"].columns:
            raise ValueError(f"cannot color by {color_by}: no sharpness.csv column")
        return interp_sharpness(run_data["sharp"], steps, color_by)
    m = run_data["metrics"]
    if color_by in m.columns:
        idx = np.searchsorted(m["step"].to_numpy(), steps)
        idx = np.clip(idx, 0, len(m) - 1)
        return m[color_by].to_numpy(float)[idx]
    if color_by == "step":
        return steps.astype(float)
    raise ValueError(f"unknown color_by '{color_by}'")


def pc_speed(coords: np.ndarray) -> np.ndarray:
    """‖Δθ‖ in PC space between consecutive snapshots (trajectory 'speed')."""
    return np.linalg.norm(np.diff(coords, axis=0), axis=1)


def delta_vtop_alignment(run_data: dict) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """|cos(Δθ_t, v_top_t)| at each curvature measurement step.

    Uses the SINGLE-step Δθ persisted at sharpness steps (dthetas.npz) — the
    correct quantity for the Damian et al. (2023) self-stabilization test.
    Falls back to multi-step snapshot diffs only if single-step Δθ is absent
    (then the result reflects slow drift, not the period-2 EoS oscillation).
    Returns (steps, |cos|) or None.
    """
    if run_data["vtops"] is None:
        return None
    V = run_data["vtops"]["V"].astype(np.float64)
    vsteps = run_data["vtops"]["steps"].astype(int)

    if run_data.get("dthetas") is not None:
        D = run_data["dthetas"]["D"].astype(np.float64)
        dsteps = run_data["dthetas"]["steps"].astype(int)
        # match each v_top to the Δθ at the same step (they were saved together)
        dmap = dict(zip(dsteps.tolist(), D))
        angles = []
        for k, vs in enumerate(vsteps):
            d = dmap.get(int(vs))
            if d is None:
                angles.append(np.nan); continue
            dn = np.linalg.norm(d); vn = np.linalg.norm(V[k])
            angles.append(abs(float(d @ V[k]) / (dn * vn)) if dn > 1e-12 and vn > 1e-12 else np.nan)
        return np.asarray(vsteps), np.asarray(angles)

    # legacy fallback: 10-step snapshot diffs (cancels EoS oscillation — less meaningful)
    W = run_data["W"]; snap_steps = run_data["snap_steps"]
    angles = []
    for k, vs in enumerate(vsteps):
        i = int(np.searchsorted(snap_steps, vs))
        i = min(max(i, 1), len(snap_steps) - 1)
        dtheta = W[i] - W[i - 1]
        dn = np.linalg.norm(dtheta); vn = np.linalg.norm(V[k])
        angles.append(abs(float(dtheta @ V[k]) / (dn * vn)) if dn > 1e-12 and vn > 1e-12 else np.nan)
    return np.asarray(vsteps), np.asarray(angles)
