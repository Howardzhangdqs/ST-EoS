"""X-SAM ablation aggregation + figure (multi-seed CIs).

Reads the per-seed CSVs figures/xsam_summary_s{0..4}.csv produced by
xsam_experiment.py and builds the §5 ablation figure:

  figures/xsam_component_bars.pdf (vector PDF, mathtext labels)
  figures/xsam_summary.csv       (mean ± 95% CI per opt)

Falsification verdict (§5): X-SAM escapes 2/η ONLY via its global SAM component.
  * xsam_full ≈ sam (overlapping λ_max_final CIs)  ⇒ the rank-1 top-eigenvector
    correction adds NOTHING to the full method (it is inert: SAM keeps
    λ_max < 2/η, so the self-gating clamp divisor = max(λ̂/(2/η), 1) never fires).
  * xsam_rank1only does NOT escape (λ_max_final ≫ 2/η ≫ 21)  ⇒ the rank-1
    correction ALONE cannot escape — the same spectral-migration failure as our
    rank-1 EoS-Precond (dampening ONE Hessian eigendir just migrates sharpness;
    here it cannot even hold the EoS equilibrium because the top eigenvector is
    orthogonal to the gradient, so the correction is a near-no-op that lets the
    trajectory drift to a sharper basin).
  Together: escape is ENTIRELY the global SAM component. §5 prediction CONFIRMED.

Usage: python scripts/analysis/analyze_xsam.py   (after running xsam_experiment.py for s=0..4)
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import setup, savefig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

setup(scale=1.1518)  # R31: 1/s, s=0.8682 (0.95\linewidth / 433.4pt)
FIG = Path("figures")

# t-distribution 95% multipliers (small-n, two-sided) — matches analyze_multiseed_control.ci()
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}

# bar order + colors: rank-1-only and sgd GRAY; sam and xsam_full GREEN
OPTS = ["sgd", "xsam_rank1only", "sam", "xsam_full"]
COLORS = dict(sgd="tab:gray", xsam_rank1only="tab:gray",
              sam="tab:green", xsam_full="tab:green")
LABELS = dict(sgd=r"SGD", xsam_rank1only=r"X-SAM rank-1 only",
              sam=r"SAM (global)", xsam_full=r"X-SAM full (SAM + rank-1)")


def ci(xs):
    xs = np.asarray([x for x in xs if x == x], float)  # drop NaN
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    return float(xs.mean()), T.get(n, 1.96) * float(xs.std(ddof=1)) / sqrt(n)


def collect():
    rows = []
    for s in range(5):
        p = FIG / f"xsam_summary_s{s}.csv"
        if p.exists():
            rows.append(pd.read_csv(p).assign(seed=s))
    if not rows:
        raise SystemExit(f"no figures/xsam_summary_s*.csv found; run xsam_experiment.py first")
    return pd.concat(rows, ignore_index=True)


def main():
    df = collect()
    n_seeds = df["seed"].nunique()
    print(f"=== X-SAM ablation (mean ± 95% CI over {n_seeds} seeds) ===")

    # per-opt mean ± CI for every numeric column
    agg = {}
    for opt in OPTS:
        sub = df[df.opt == opt]
        agg[opt] = {col: ci(sub[col]) for col in
                    ["final_loss", "min_loss", "lambda_max_final", "lambda_max_max"]}
        lam_m, lam_c = agg[opt]["lambda_max_final"]
        fl_m, fl_c = agg[opt]["final_loss"]
        print(f"  {opt:16s} λ_max_final = {lam_m:7.2f} ± {lam_c:5.2f}   "
              f"final_loss = {fl_m:.4f} ± {fl_c:.4f}")

    # ---- verdict (the §5 falsification) --------------------------------------
    TWO_OVER_ETA = 40.0
    print(f"\n=== §5 ABLATION VERDICT (2/η = {TWO_OVER_ETA:g}) ===")
    sam_m, sam_c = agg["sam"]["lambda_max_final"]
    xf_m, xf_c = agg["xsam_full"]["lambda_max_final"]
    xr_m, xr_c = agg["xsam_rank1only"]["lambda_max_final"]
    sgd_m, sgd_c = agg["sgd"]["lambda_max_final"]

    # CI overlap test (xsam_full vs sam)
    xf_xu, xf_xl = xf_m + xf_c, xf_m - xf_c
    sam_xu, sam_xl = sam_m + sam_c, sam_m - sam_c
    overlap = not (xf_xl > sam_xu or sam_xl > xf_xu)
    print(f"  xsam_full vs SAM    : {xf_m:.2f}±{xf_c:.2f}  vs  {sam_m:.2f}±{sam_c:.2f}  "
          f"→ CIs {'OVERLAP' if overlap else 'DISJOINT'} ⇒ "
          f"rank-1 {'adds NOTHING to escape' if overlap else 'CONTRIBUTES'} (§5 "
          f"{'CONFIRMED' if overlap else 'WEAKENED'})")
    # does rank-1-only escape? (escape = λ_max_final well below 2/η, ~SAM level)
    rank1_escapes = xr_m + xr_c < 0.7 * TWO_OVER_ETA
    rank1_at_threshold = (xr_m - xr_c) < 1.3 * TWO_OVER_ETA and (xr_m + xr_c) > 0.7 * TWO_OVER_ETA
    print(f"  xsam_rank1only      : {xr_m:.2f}±{xr_c:.2f}  (2/η={TWO_OVER_ETA:g})  "
          f"→ {'ESCAPES (unexpected!)' if rank1_escapes else 'does NOT escape'} "
          f"({'locks ≈ 2/η' if rank1_at_threshold else 'λ_max ≫ 2/η (spectral-migration drift)'})")
    print(f"  SGD reference       : {sgd_m:.2f}±{sgd_c:.2f}  (locks at 2/η)")

    if overlap and not rank1_escapes:
        print("\n  ⇒ Escape is ENTIRELY the global SAM component. The rank-1 top-")
        print("    eigenvector correction contributes nothing to escape (it is inert")
        print("    under SAM, and alone it suffers the spectral-migration failure).")
        print("    §5 prediction CONFIRMED.")

    # ---- CSV summary ---------------------------------------------------------
    out_rows = []
    for opt in OPTS:
        for col in ["final_loss", "min_loss", "lambda_max_final", "lambda_max_max"]:
            m, c = agg[opt][col]
            out_rows.append(dict(opt=opt, metric=col, mean=m, ci95=c))
    pd.DataFrame(out_rows).to_csv(FIG / "xsam_summary.csv", index=False)
    print(f"\nsaved figures/xsam_summary.csv")

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    means = [agg[o]["lambda_max_final"][0] for o in OPTS]
    errs = [agg[o]["lambda_max_final"][1] for o in OPTS]
    cols = [COLORS[o] for o in OPTS]
    xpos = np.arange(len(OPTS))
    ax.bar(xpos, means, yerr=errs, capsize=5, color=cols, alpha=0.85,
           edgecolor="0.3", width=0.62)
    ax.axhline(TWO_OVER_ETA, color="tab:red", ls="--", lw=1.3,
               label=r"$2/\eta = 40$ (EoS threshold)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[o] for o in OPTS], rotation=15, ha="right")
    ax.set_ylabel(r"$\lambda_{\max}$ final")
    ax.set_title(r"X-SAM escape tracks the global SAM component"
                 r" (rank-1 correction adds nothing)", loc='left')
    # annotate each bar with its mean
    ymax = max(m + c for m, c in zip(means, errs) if np.isfinite(m))
    for i, (m, c) in enumerate(zip(means, errs)):
        if not np.isfinite(m):
            continue
        ax.text(i, m + c + 0.04 * ymax, f"{m:.1f}", ha="center", fontsize=9.21)
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(fontsize=9.21, loc="upper left")
    # NOT "28_xsam_ablation": that slot is the paper's two-panel multiseed
    # figure (plot_xsam_multiseed.py). Writing the same name here silently
    # clobbered it once (2026-09-24) — keep this exploration figure separate.
    out = savefig(fig, "xsam_component_bars")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
