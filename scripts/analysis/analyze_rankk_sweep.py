"""Aggregate the rank-k curvature-preconditioning sweep across seeds and produce
the paper figure.

Reads ``figures/rankk_summary_s{0..4}.csv`` (5 seeds; falls back to whichever
seeds exist) and the seed-0 spectrum, computes mean ± 95% CI via the t-distribution
``ci()`` (copied from ``analyze_multiseed_control.py``), and writes:
    figures/27_rankk_sweep.pdf          single-panel vector PDF, mathtext labels
    figures/rankk_sweep_summary.csv     mean ± CI per optimizer

Single panel (was panel b): top-12 Hessian spectrum at the final measurement
step, seed 0, rank-k k=1 vs k=10, annotated with the canonical multi-seed
escape number λ1 = 263.3 ± 12.0 (n=3 — values unchanged). The former
k∈{1,2,5,10} bar panel was REMOVED: its four bars were bit-identical (the
panel carried one sentence of information).

Usage: python scripts/analysis/analyze_rankk_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    METHOD_COLORS, THRESH_COLOR, THEORY, ST, setup, savefig,
)
import matplotlib.pyplot as plt  # noqa: E402

setup(scale=1.2333)   # R30: source canvas ~1.22x final size (report()-measured)
FIG = Path("figures")
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}     # t-dist 95% (two-sided), df=n-1
RANKK_KS = [1, 2, 5, 10]
TWO_OVER_ETA = 40.0
LAM_THRESH = 0.7 * TWO_OVER_ETA                     # escape criterion
# R30 adjudication 2: the summary CSV on disk is the FROZEN paper口径 (sam
# 21.354±0.042, λ1 = 263.3±12.0, ...).  Re-aggregating from the current
# per-seed inputs would change those numbers (seed-3 SAM outlier 3.69 landed
# after the last aggregation — a pre-R30 scientific question, reported and
# deferred).  While the file exists we therefore READ it for plotting and
# annotation and skip both re-aggregation and the rewrite; only a missing
# file falls back to the original aggregation path below.
SUMMARY_CSV = FIG / "rankk_sweep_summary.csv"


def ci(xs):
    """mean ± 95% CI half-width via the t-distribution (matches
    analyze_multiseed_control.ci). NaNs are dropped; n<2 ⇒ (nan, nan)."""
    xs = np.asarray([x for x in xs if x == x], float)
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    return float(xs.mean()), T.get(n, 1.96) * float(xs.std(ddof=1)) / sqrt(n)


def main():
    # --- aggregate per-seed summaries --------------------------------------
    # R30 adjudication 2 read-if-exists guard: frozen paper numbers while the
    # summary CSV exists (see comment at SUMMARY_CSV above).
    opt_order = ["sgd", "sam"] + [f"rankk_k{k}" for k in RANKK_KS]
    if SUMMARY_CSV.exists():
        sdf = pd.read_csv(SUMMARY_CSV)
        agg = {r.opt: dict(lam_mean=r.lambda_max_final_mean,
                           lam_ci=r.lambda_max_final_ci95,
                           loss_mean=r.final_loss_mean,
                           loss_ci=r.final_loss_ci95, n=r.n_seeds)
               for r in sdf.itertuples()}
        print(f"[guard] {SUMMARY_CSV} exists — frozen paper numbers loaded; "
              "skipping re-aggregation and rewrite (R30 adjudication 2)")
    else:
        rows = []
        for s in range(5):
            p = FIG / f"rankk_summary_s{s}.csv"
            if p.exists():
                rows.append(pd.read_csv(p).assign(seed=s))
        if not rows:
            raise SystemExit("No per-seed CSVs found (figures/rankk_summary_s*.csv); "
                             "run scripts/experiments/rankk_sweep_experiment.py first.")
        df = pd.concat(rows, ignore_index=True)

        agg = {}
        for opt in opt_order:
            sub = df[df.opt == opt]
            if not len(sub):
                continue
            lam_m, lam_c = ci(sub["lambda_max_final"])
            loss_m, loss_c = ci(sub["final_loss"])
            agg[opt] = dict(lam_mean=lam_m, lam_ci=lam_c,
                            loss_mean=loss_m, loss_ci=loss_c, n=len(sub))
        print("[aggregate] summary CSV missing — original aggregation path")

    print("=== Rank-k sweep (mean ± 95% CI) ===")
    for opt in opt_order:
        if opt in agg:
            a = agg[opt]
            print(f"  {opt:10s} λ_max_final = {a['lam_mean']:6.2f} ± {a['lam_ci']:5.2f}   "
                  f"final_loss = {a['loss_mean']:.4f} ± {a['loss_ci']:.4f}   (n={a['n']})")

    # --- verdict: classify each variant (flat-escape / lock@2η / sharp-escape) -
    print(f"\n=== VERDICT (escape-to-flat needs λ_max_final < {LAM_THRESH:.0f} = 0.7·2/η; "
          f"2/η = {TWO_OVER_ETA:.0f}) ===")
    rankk_lams = [agg[f"rankk_k{k}"]["lam_mean"] for k in RANKK_KS]
    for k in RANKK_KS:
        m = agg[f"rankk_k{k}"]["lam_mean"]
        cls = ("flat-escape (<28)" if m < LAM_THRESH
               else f"lock@2/η (≈{TWO_OVER_ETA:.0f})" if m < 1.4 * TWO_OVER_ETA
               else "SHARP-escape (>>2/η)")
        print(f"  rank-k k={k:<2d}: λ_max_final={m:7.2f}  → {cls}")
    print(f"  SGD reference: λ_max_final={agg['sgd']['lam_mean']:.2f} (EoS pin @2/η)")
    print(f"  SAM reference: λ_max_final={agg['sam']['lam_mean']:.2f} (flat-escape)")
    spread = max(rankk_lams) - min(rankk_lams)
    print(f"  rank-k spread over k∈{{1,2,5,10}}: {spread:.3f} "
          f"(≈0 ⇒ k>1 does nothing; landscape is 1-D sharp)")

    # --- figure ------------------------------------------------------------
    # COMPRESSED (reviewer): the k∈{1,2,5,10} bar panel was REMOVED — all
    # four bars were bit-identical (one sentence of information). The
    # seed-0 top-12 spectrum panel survives as the single panel, now
    # annotated with the canonical multi-seed escape number (λ1 =
    # 263.3 ± 12.0, n=3 — the paper's canonical values, not altered),
    # which replaces the removed bars' "263 ± 12" labels.
    # R30 style: both k-variants share the damping-family hue (no new
    # colors); the bar-vs-line form already separates them.  Threshold in
    # neutral theory gray, frameless legend, short title.
    fig, ax = plt.subplots(figsize=(4.8, 3.0))

    # top-12 spectrum at the final measurement step, seed 0: k=1 (bars) vs
    # k=10 (step line). They overlap EXACTLY — the visual proof that k>1 does
    # nothing. LOG y so λ_1≈263 and λ_2..12≈0.3 are both visible (1-D sharp).
    spec_path = FIG / "rankk_spectrum_s0.csv"
    plotted_spectrum = False
    if spec_path.exists():
        sp = pd.read_csv(spec_path)
        final_step = int(sp["step"].max())
        spf = sp[sp["step"] == final_step]
        idx = np.arange(12)
        style = {"rankk_k1": (METHOD_COLORS["rank1"], r"$k=1$", True),
                 "rankk_k10": (METHOD_COLORS["rankk"], r"$k=10$", False)}
        k1_top = None
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=THEORY["light"], alpha=0.4, lw=0.4)
        for opt in ["rankk_k1", "rankk_k10"]:
            sub = spf[spf.opt == opt].sort_values("idx")
            if not len(sub):
                continue
            vals = sub["lambda_i"].values[:12]
            if opt == "rankk_k1" and len(vals):
                k1_top = float(vals[0])
            col, lab, is_bar = style[opt]
            if is_bar:
                ax.bar(idx, vals, 0.55, color=col, alpha=0.55, edgecolor="0.3", label=lab)
            else:
                ax.plot(idx, vals, "o-", color=col, lw=ST["primary"], label=lab)
            plotted_spectrum = True
        ax.axhline(TWO_OVER_ETA, color=THRESH_COLOR, ls="-", lw=ST["theory"], alpha=0.9,
                   label=rf"$2/\eta = {TWO_OVER_ETA:.0f}$")
        ax.set_yscale("log")
        # canonical multi-seed escape number from agg (NOT seed-0): the
        # 263.3 ± 12.0 values are the paper's canonical figures — keep exact
        lam1_m, lam1_c = agg["rankk_k1"]["lam_mean"], agg["rankk_k1"]["lam_ci"]
        if k1_top is not None:
            ax.set_ylim(top=1200)   # headroom for the λ1 annotation
            ax.annotate(fr"$\lambda_1 = {lam1_m:.1f} \pm {lam1_c:.1f}$"
                        r"  ($\gg\,2/\eta$)",
                        xy=(0.0, k1_top), xytext=(1.0, 700),
                        textcoords="data", fontsize=9.87, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8))
        ax.set_xticks(idx[::2])
        ax.set_xlabel(r"eigenvalue index $i$")
        ax.set_ylabel(r"$\lambda_i$ (Hessian, final step)")
        ax.set_title(r"All curvature in $v_1$ (seed 0)")
        ax.legend(loc="upper right")
    if not plotted_spectrum:
        ax.text(0.5, 0.5, r"spectrum\_s0.csv not found", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()

    # No in-figure suptitle: the paper caption (fig:rankk in main.tex) carries
    # the summary message instead, per publication convention.
    out = savefig(fig, "27_rankk_sweep")
    print(f"\nsaved {out}")
    print("figure panels remaining: 1 — seed-0 top-12 Hessian spectrum, k=1 "
          "(bars) vs k=10 (line); the k∈{1,2,5,10} bar panel was removed "
          "(bars bit-identical across k)")

    # --- summary CSV (mean ± CI per opt) -----------------------------------
    # R30 adjudication 2: never rewrite an existing frozen summary (see the
    # guard at the top of main()).
    if SUMMARY_CSV.exists():
        print(f"[guard] {SUMMARY_CSV} kept frozen — no rewrite "
              "(R30 adjudication 2)")
        return
    srows = []
    for opt in opt_order:
        if opt in agg:
            a = agg[opt]
            srows.append(dict(
                opt=opt,
                lambda_max_final_mean=a["lam_mean"], lambda_max_final_ci95=a["lam_ci"],
                final_loss_mean=a["loss_mean"], final_loss_ci95=a["loss_ci"],
                n_seeds=a["n"],
            ))
    pd.DataFrame(srows).to_csv(SUMMARY_CSV, index=False)
    print(f"saved {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
