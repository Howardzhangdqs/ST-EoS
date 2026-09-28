"""Gap B — is the output-layer weight norm specifically a sharpness proxy?

Li et al. (NeurIPS 2022) report that the OUTPUT-layer weight norm tracks the
sharpness along the GD trajectory. The decisive, falsifiable test is: is
‖W_out‖ a BETTER predictor of λ_max than the hidden-layer norms ‖W_hidden‖?
If every layer's norm correlates equally with λ_max, the "output layer"
specificity claim weakens; if ‖W_out‖ is consistently on top, the claim holds.

We already have, for every run, per-step ‖W‖ for each Linear layer (metrics.csv)
and λ_max at intervals (sharpness.csv). This script interpolates λ_max to every
step and computes Pearson + Spearman correlation between each layer's weight
norm and λ_max along the trajectory, across ALL runs under runs/.

Usage:
    python scripts/experiments/gap_b_proxy.py --run-root runs --out figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.analysis.trajectory import interp_sharpness


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    am, bm = a.mean(), b.mean()
    da, db = a - am, b - bm
    denom = np.sqrt((da ** 2).sum() * (db ** 2).sum())
    return float((da @ db) / denom) if denom > 0 else float("nan")


def _spearman(a, b):
    return float(spearmanr(a, b).correlation)


def _layer_norm_cols(metrics: pd.DataFrame) -> list[str]:
    """Linear-layer weight-norm columns in forward order, output layer LAST.
    e.g. ['wnorm_fc1','wnorm_fc2','wnorm_fc_out']."""
    cols = [c for c in metrics.columns if c.startswith("wnorm_fc") or c.startswith("wnorm_l")]
    # drop the alias 'wnorm_output' if present; it duplicates fc_out
    cols = [c for c in cols if c != "wnorm_output"]
    # ensure deterministic order: fc1, fc2, ..., fc_out
    def key(c):
        n = c.replace("wnorm_", "").replace("fc", "").replace("l", "")
        return (0, int(n)) if n.isdigit() else (1, c)
    return sorted(cols, key=key)


def per_run(run_dir: Path) -> dict | None:
    m_path = run_dir / "metrics.csv"
    s_path = run_dir / "sharpness.csv"
    if not (m_path.exists() and s_path.exists()):
        return None
    m = pd.read_csv(m_path)
    sh = pd.read_csv(s_path)
    if "lambda_max" not in sh.columns:
        return None
    steps = m["step"].to_numpy(float)
    lam = interp_sharpness(sh, steps, "lambda_max")
    rec = {"run": run_dir.name, "parent": run_dir.parent.name}
    for c in _layer_norm_cols(m):
        w = m[c].to_numpy(float)
        mask = np.isfinite(w) & np.isfinite(lam)
        if mask.sum() >= 10:
            rec[f"{c}__pearson"] = _pearson(w[mask], lam[mask])
            rec[f"{c}__spearman"] = _spearman(w[mask], lam[mask])
        else:
            rec[f"{c}__pearson"] = float("nan")
            rec[f"{c}__spearman"] = float("nan")
    rec["output_col"] = _layer_norm_cols(m)[-1]
    rec["n_steps_finite"] = int(np.isfinite(lam).sum())
    return rec


def collect(run_root: Path) -> pd.DataFrame:
    rows = []
    for mp in sorted(run_root.rglob("metrics.csv")):
        r = per_run(mp.parent)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = collect(Path(args.run_root))
    if df.empty:
        print("no runs with metrics.csv + sharpness.csv found"); return

    # pick the common layer columns (intersection across runs)
    layer_cols = sorted({c.rsplit("__", 1)[0] for c in df.columns if c.endswith("__pearson")},
                        key=lambda c: (0, int(c.replace("wnorm_fc", ""))) if c.replace("wnorm_fc", "").isdigit() else (1, c))
    print(f"\nLayers analysed (forward order): {layer_cols}")
    print(f"Output layer = {df['output_col'].iloc[-1]}\n")

    # ---- table: mean Pearson/Spearman per layer across runs ----
    summary = []
    for c in layer_cols:
        p = df[f"{c}__pearson"].dropna()
        s = df[f"{c}__spearman"].dropna()
        summary.append(dict(layer=c, n_runs=len(p),
                            pearson_mean=p.mean(), pearson_min=p.min(), pearson_max=p.max(),
                            spearman_mean=s.mean()))
    summ = pd.DataFrame(summary)
    print("=== per-layer correlation of ‖W‖ with λ_max along trajectories ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summ.to_csv(out / "gapB_layer_correlation_summary.csv", index=False)
    df.to_csv(out / "gapB_per_run_correlations.csv", index=False)

    is_out = [c == df["output_col"].iloc[-1] for c in layer_cols]

    # ---- Figure A: grouped bars — corr per layer across runs ----
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.8 / len(layer_cols)
    for i, c in enumerate(layer_cols):
        vals = df[f"{c}__pearson"].to_numpy(float)
        col = "tab:red" if c == df["output_col"].iloc[-1] else f"tab:{['blue','green','orange','purple'][i % 4]}"
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=c, color=col, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson corr(‖W_layer‖, λ_max)")
    ax.set_title("Gap B: is the OUTPUT-layer norm a better λ_max proxy than hidden layers?  (red = output)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(out / "10_gapB_layer_corr_bars.png", dpi=140); plt.close(fig)

    # ---- Figure B: scatter ‖W_out‖ vs λ_max for the canonical EoS run ----
    canonical = Path(args.run_root) / "mlp_ts_eos"
    if canonical.exists():
        m = pd.read_csv(canonical / "metrics.csv")
        sh = pd.read_csv(canonical / "sharpness.csv")
        steps = m["step"].to_numpy(float)
        lam = interp_sharpness(sh, steps, "lambda_max")
        wout = m["wnorm_output"].to_numpy(float)
        mask = np.isfinite(wout) & np.isfinite(lam)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(wout[mask], lam[mask], c=steps[mask], cmap="viridis", s=10)
        ax.set_xlabel("‖W_out‖  (output-layer weight norm)")
        ax.set_ylabel("λ_max  (interpolated)")
        ax.set_title(f"Gap B — output-norm vs sharpness  (mlp_ts_eos, "
                     f"Pearson={_pearson(wout[mask], lam[mask]):.3f})")
        fig.colorbar(sc, label="step")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "11_gapB_wout_vs_lambdamax.png", dpi=140); plt.close(fig)

    out_col = df["output_col"].iloc[-1]
    hid_cols = [c for c in layer_cols if c != out_col]
    out_p = df[f"{out_col}__pearson"].mean()
    hid_p = np.mean([df[f"{c}__pearson"].mean() for c in hid_cols])
    print(f"\nmean Pearson: output({out_col})={out_p:.3f}  vs  hidden-mean={hid_p:.3f}")
    print(f"=> output layer {'IS' if out_p > hid_p + 0.05 else 'is NOT clearly'} a sharper proxy "
          f"(Δ={out_p-hid_p:+.3f})")


if __name__ == "__main__":
    main()
