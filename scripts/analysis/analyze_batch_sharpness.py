"""Batch sharpness proper vs minibatch λmax — review round 3 (§ stochastic EoS).

The EoSS literature (Andreyev & Beneventano 2024; +momentum 2026) defines the
*batch sharpness* as the expected directional curvature along the batch
gradient,  λ_step = E_B[ g_Bᵀ H_B g_B / ‖g_B‖² ],  NOT the top eigenvalue of
the minibatch Hessian.  The revision-2 cross-batch study measured the latter
(lam1_curmb); the mb2 rerun adds the former:

  lhg_curmb   gradient Rayleigh quotient on the step's own current minibatch
  lhg_bs_mean/std  m=4-fresh-minibatch average of the same quantity
                  (the direct E_B[·] estimate)

This script produces the comparison the review asked for (late window,
step ≥ 2000, mean ± 95% CI over seeds):

  fixed reference λmax (seed123) | fixed b124 / b125 λmax
  current-minibatch λmax  | current-minibatch RQ  | E_B batch sharpness
against the GD boundary 2/η = 20 and the heavy-ball reference 2(1+μ)/η = 38,
plus per-quantity fractions above/below 38, and a trajectory figure.

Reads cifar100_r18_sgd_detarpack_mb2_s*  (--prefix to point at the legacy
mb runs, which lack the lhg_* columns — they are skipped gracefully).
Outputs: figures/batch_sharpness_summary.csv + figures/40_batch_sharpness.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steos.analysis.figure_style import C, THEORY, ST, setup, savefig
import numpy as np
import pandas as pd
from scipy import stats

LATE_STEP = 2000
ETA, MU = 0.1, 0.9
HB = 2 * (1 + MU) / ETA          # 38 — heavy-ball linear-stability reference
GD = 2 / ETA                     # 20 — full-batch GD boundary


def ci95(xs):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) < 2:
        return float(xs.mean()) if len(xs) else float("nan"), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_ci(delta):
    """mean and 90% CI of per-seed paired differences (t, n seeds)."""
    d = np.asarray(delta, float)
    n = len(d)
    if n < 2:
        return float(d.mean()), 0.0, float(d.mean()), 0.0
    m = float(d.mean())
    h90 = float(stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n))
    return m, h90, m - h90, m + h90


def main():
    # R30: oversized source canvas shrunk at \includegraphics; scale keeps
    # final on-paper fonts/strokes at the R30 baseline (report()-measured).
    setup(scale=1.1784)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cifar100_r18_sgd_detarpack_mb2_s")
    ap.add_argument("--late-step", type=int, default=LATE_STEP)
    args = ap.parse_args()

    from pathlib import Path
    dfs = []
    for d in sorted(Path("runs").glob(args.prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            dfs.append((int(s), pd.read_csv(d / "metrics.csv")))
    if not dfs:
        raise SystemExit(f"no runs found for prefix {args.prefix}")
    print(f"runs: seeds {[s for s, _ in dfs]}  (late window step ≥ {args.late_step})")

    quantities = [  # (label, column, kind)
        ("$\\lambda_{\\mathrm{ref}}$ fixed eval batch (seed 123)", "lambda_max", "lam"),
        ("$\\lambda_{\\mathrm{ref}}$ fixed b124", "lam1_b124", "lam"),
        ("$\\lambda_{\\mathrm{ref}}$ fixed b125", "lam1_b125", "lam"),
        ("$\\lambda_{\\max}$ current minibatch", "lam1_curmb", "lam"),
        ("RQ current minibatch ($\\hat\\lambda_{\\mathrm{step}}$)", "lhg_curmb", "rq"),
        ("batch sharpness $\\mathrm{E}_B[m{=}4]$", "lhg_bs_mean", "rq"),
    ]

    rows, per_seed_late = [], {}
    for label, col, kind in quantities:
        if col not in dfs[0][1].columns:
            print(f"[skip] column {col} absent (legacy run?)")
            continue
        per_seed, pooled = [], []
        for _, df in dfs:
            m = df[df["step"] >= args.late_step]
            per_seed.append(float(m[col].mean()))
            pooled.append(m[col].to_numpy())
        per_seed_late[label] = per_seed
        m, h = ci95(per_seed)
        allv = np.concatenate(pooled)
        allv = allv[~np.isnan(allv)]
        frac_above = float((allv > HB).mean())
        rows.append(dict(quantity=label, mean=m, ci95=h, n_seeds=len(per_seed),
                         median=float(np.median(allv)),
                         iqr_lo=float(np.percentile(allv, 25)),
                         iqr_hi=float(np.percentile(allv, 75)),
                         frac_above_hb=frac_above, frac_below_hb=1 - frac_above))
        star = " <-- EoSS quantity" if kind == "rq" else ""
        print(f"{label:34s} late {m:6.2f} ± {h:4.2f}   median {np.median(allv):6.2f} "
              f"IQR [{np.percentile(allv,25):.1f},{np.percentile(allv,75):.1f}] "
              f"frac>38 {frac_above:.2f}{star}")

    # paired contrasts: E_B batch sharpness vs minibatch λmax (same seeds)
    if "batch sharpness E_B[m=4]" in per_seed_late and "λmax current minibatch" in per_seed_late:
        d = np.array(per_seed_late["batch sharpness E_B[m=4]"]) - \
            np.array(per_seed_late["λmax current minibatch"])
        m, h, lo, hi = paired_ci(d)
        rows.append(dict(quantity="paired Δ(E_B − λmax curmb)", mean=m, ci95=h,
                         n_seeds=len(d), median=float("nan"), iqr_lo=lo, iqr_hi=hi,
                         frac_above_hb=float("nan"), frac_below_hb=float("nan")))
        print(f"paired Δ(E_B − λmax_curmb) = {m:+.2f}  90% CI [{lo:+.2f},{hi:+.2f}]")

    # R30 zero-change guard: never rewrite an existing summary CSV
    # (authorized minimal IO guard, no analysis-logic change).
    csv_path = Path("figures/batch_sharpness_summary.csv")
    if csv_path.exists():
        print(f"kept existing {csv_path} (rewrite skipped)")
    else:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print("=> figures/batch_sharpness_summary.csv")

    # ---- figure: trajectories (seed-mean ± min/max band) ----
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2))
    # R30 palette mapping (quantity semantics, registry hues only): the
    # canonical fixed reference -> theory gray; the revision-2 quantity
    # (current-minibatch top eigenvalue) -> neutral slate; the RQ variant
    # of the E_B family -> damping blue; the EoSS batch sharpness (the
    # subject of the comparison) -> sam teal.
    colors = {"$\\lambda_{\\mathrm{ref}}$ fixed eval batch (seed 123)": C["theory"],
              "$\\lambda_{\\max}$ current minibatch": C["neutral"],
              "RQ current minibatch ($\\hat\\lambda_{\\mathrm{step}}$)": C["damping"],
              "batch sharpness $\\mathrm{E}_B[m{=}4]$": C["sam"]}
    # R39 (fixplan §5.8/§6): redundant linestyle+marker coding for the four
    # data series (grayscale-printable); markers ride the bold seed-mean only.
    series_style = {
        "$\\lambda_{\\mathrm{ref}}$ fixed eval batch (seed 123)":
            dict(linestyle="-", marker="o"),
        "$\\lambda_{\\max}$ current minibatch":
            dict(linestyle="--", marker="s"),
        "RQ current minibatch ($\\hat\\lambda_{\\mathrm{step}}$)":
            dict(linestyle="-.", marker="^"),
        "batch sharpness $\\mathrm{E}_B[m{=}4]$":
            dict(linestyle=(0, (5, 2, 1, 2)), marker="D"),
    }
    for label, c in colors.items():
        if label not in [q[0] for q in quantities if q[1] in dfs[0][1].columns]:
            continue
        col = dict((q[0], q[1]) for q in quantities)[label]
        steps = dfs[0][1]["step"].to_numpy()
        arrs = np.stack([df[col].to_numpy() for _, df in dfs
                         if col in df.columns])
        mm, lo_, hi_ = arrs.mean(0), arrs.min(0), arrs.max(0)
        st = series_style[label]
        for ax in axes:
            ax.plot(steps, mm, color=c, lw=ST["primary"], label=label,
                    linestyle=st["linestyle"], marker=st["marker"],
                    markevery=max(1, len(steps) // 10),
                    markersize=ST["marker"], markerfacecolor=c,
                    markeredgecolor=c)
            ax.fill_between(steps, lo_, hi_, color=c, alpha=ST["alpha_ci"],
                            lw=0)
    axes[0].set_yscale("log")
    axes[0].set_title(r"\textbf{(a)} Full trajectory (log)", color="#1A1D21")
    axes[1].axvline(args.late_step, color=THEORY["light"], ls=":", lw=ST["theory"])
    axes[1].set_ylim(0, 80)
    axes[1].set_title(r"\textbf{(b)} Boundary zoom", color="#1A1D21")
    for i, ax in enumerate(axes):
        # R30: boundary references at theory weight/neutral dark gray.
        # R38 (user): the two reference lines enter the legend (own row
        # below the curve row) instead of carrying in-panel text labels —
        # label only the first panel's lines to avoid duplicate handles.
        hb_kw = (dict(label=rf"HB ref. $2(1{{+}}\mu)/\eta = {HB:.0f}$") if i == 0
                 else {})
        gd_kw = (dict(label=rf"$2/\eta = {GD:.0f}$") if i == 0 else {})
        ax.axhline(HB, color=C["theory"], ls="--", lw=ST["theory"], **hb_kw)
        ax.axhline(GD, color=C["theory"], ls=":", lw=ST["theory"], **gd_kw)
        ax.set_xlabel("step")
        # trajectory panels carry no grid (R30 rule)
    # R39 (fixplan §5.8/§6): ONE figure-level legend below the panels, ncol=3
    # over two rows (row 1: the four quantity curves; row 2: the two
    # reference lines). fontsize 9.45 pt on the source canvas: the legend
    # block measures ~427 pt wide (probe-measured), inside the ~441 pt
    # panel-block width, so the tight page width (444.3 pt) and hence the
    # inclusion magnification s=0.847 are unchanged -> every OTHER font keeps
    # its R30/R38 final size, and the legend lands at 9.45*0.847 = 8.0 pt
    # final (the fixplan-mandated raise from the old 6.6 pt outlier; the
    # only font-size increase allowed in this figure).
    hnd, lab = axes[0].get_legend_handles_labels()
    # R43 (user): halve the visible gap between the legend block and the
    # xlabel band above it.  Anchor y is a pure vertical translation of the
    # legend (tight crop follows the legend, everything else fixed), so
    # -0.02 -> 0.049 moves the block up 0.069 x 158.4 pt ~= 11 pt, which
    # halves the pixel-measured ink gap 21.96 pt -> ~11 pt.  Only the y
    # coordinate changed; x (centering), ncol, fonts, spacing untouched.
    fig.legend(hnd, lab, loc="upper center", bbox_to_anchor=(0.5, 0.049),
               ncol=3, frameon=False, fontsize=9.45,
               handlelength=2.2, columnspacing=0.9, labelspacing=0.35)
    axes[0].set_ylabel("sharpness")
    # R30: the long conclusion/protocol suptitle (SGD, ResNet-18/CIFAR-100,
    # formula) moved to the paper caption; short tagged panel titles above.
    fig.tight_layout()
    out_pdf = savefig(fig, "40_batch_sharpness")
    print(f"=> {out_pdf}")


if __name__ == "__main__":
    main()
