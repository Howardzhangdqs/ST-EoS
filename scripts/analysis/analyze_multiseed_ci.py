"""Multi-seed confidence-interval aggregation for Paper A headline numbers.

Computes mean +/- 95% CI across seeds for the F20/F14 proxy correlations
(CIFAR-ResNet momentum+BN runs) and emits a publication-quality PDF figure with
LaTeX-rendered math (project standard: steos.analysis.figure_style).

Currently covers F20/F14 (cifar_rayleigh seeds). F19/F12/F15 aggregates fold in
once those multi-seed runs exist (their per-seed summary CSVs).

Usage: python scripts/analysis/analyze_multiseed_ci.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    setup, C, THEORY, ST, panel_label,
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# R30 aesthetic redesign (aesthetic_diagnosis_r29.md p4, 4.3 -> 7.5):
# true 2x2 equal-width grid (was an asymmetric "two large + two small"
# dashboard), all four panels dot-and-whisker instead of bars, focal
# estimator in damping blue, comparators in slate/light gray, faint
# horizontal grid, panel_label short titles. Numbers/CIs/values unchanged.
SCALE = 1.4302
setup(scale=SCALE)
RUNS = Path("runs")
T975 = {3: 4.303, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}  # t(df,0.025)


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float(a @ b / d) if d > 0 else float("nan")


def detrend(a, w=5):
    a = np.asarray(a, float); k = np.ones(w) / w
    return a - np.pad(np.convolve(a, k, "same"), (0, 0), mode="edge")


def ci95(xs):
    xs = np.asarray([x for x in xs if x == x], float)
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), n
    t = T975.get(n - 1, 1.96)
    return float(xs.mean()), t * float(xs.std(ddof=1)) / sqrt(n), n


def eos_block(m, col, al, thr=0.7):
    lam = m["lambda_max"].to_numpy(float)
    f = np.isfinite(lam) & (lam > 1e-3)
    l = lam[f]
    whole = pear(detrend(m[col].to_numpy(float)[f]), detrend(l))
    eos = np.abs(m[al].to_numpy(float)[f]) > thr
    if eos.sum() > 5:
        ve, le = m[col].to_numpy(float)[f][eos], l[eos]
        er = pear(detrend(ve), detrend(le))
        err = float(np.mean(np.abs(ve - le) / le))
    else:
        er, err = float("nan"), float("nan")
    return whole, er, err


def collect():
    """cifar_rayleigh seed 0..4. Returns dict[metric]->list per seed."""
    runs = ["cifar_rayleigh"] + [f"cifar_rayleigh_s{s}" for s in range(1, 5)]
    R = {k: [] for k in ["lg_whole", "lg_eos", "lg_err", "lu_whole", "lu_eos", "lu_err",
                         "bngamma_whole", "cos_g", "cos_dt"]}
    for r in runs:
        p = RUNS / r / "metrics.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p)
        wg, eg, egerr = eos_block(m, "lambda_hat_grad", "cos_grad_vtop")
        wu, eu, euerr = eos_block(m, "lambda_hat_update", "cos_update_vtop")
        lam = m["lambda_max"].to_numpy(float); f = np.isfinite(lam) & (lam > 1e-3)
        R["lg_whole"].append(wg); R["lg_eos"].append(eg); R["lg_err"].append(egerr)
        R["lu_whole"].append(wu); R["lu_eos"].append(eu); R["lu_err"].append(euerr)
        R["bngamma_whole"].append(pear(detrend(m["norm_bn_gamma"].to_numpy(float)[f]), detrend(lam[f])))
        R["cos_g"].append(float(np.abs(m["cos_grad_vtop"].to_numpy(float)[f]).mean()))
        R["cos_dt"].append(float(np.abs(m["cos_update_vtop"].to_numpy(float)[f]).mean()))
    return R


def dot_whisker(ax, xs, means, cIs, cols, marks, fills):
    """Point estimate + CI whisker, R30 hierarchy (ST marker/errbar/capsize)."""
    for x, m, c, col, mk, fs in zip(xs, means, cIs, cols, marks, fills):
        ax.errorbar([x], [m], yerr=[[c], [c]], marker=mk, ls="none", color=col,
                    fillstyle=fs, markersize=ST["marker"], markeredgewidth=ST["errbar"],
                    elinewidth=ST["errbar"], capsize=ST["capsize"], capthick=ST["errbar"],
                    markeredgecolor=col,
                    markerfacecolor=col if fs == "full" else "none")


def faint_ygrid(ax):
    """Very faint horizontal major grid (point/CI summary panels only)."""
    ax.grid(axis="y", which="major", color=THEORY["light"], alpha=0.22, lw=0.6)


def dyn_ylim(pairs, lo_pad=0.45, hi_pad=0.95, anchor=None):
    """Dynamic y-range from (mean, ci95) pairs — replaces the hard-coded
    per-panel ylims (user round: "纵坐标太固定了").

    lo = lowest whisker tip  - lo_pad × span
    hi = highest whisker tip + hi_pad × span   (hi_pad doubles as headroom
    for the value annotations, which ride 5pt above the CI tip). `anchor`
    (e.g. 0 for a signed/no-correlation baseline) is forced into the range
    before padding when given.
    """
    lo = min(m - c for m, c in pairs)
    hi = max(m + c for m, c in pairs)
    if anchor is not None:
        lo, hi = min(lo, anchor), max(hi, anchor)
    span = max(hi - lo, 1e-9)
    return lo - lo_pad * span, hi + hi_pad * span


def annotate_vals(ax, xs, means, cIs, below=None, dxs=None, dys=None):
    """Value labels 5pt above each CI tip — or, where `below` is set, directly
    below the point, centered on its x (panel (a)'s first label: the widened
    xlim (-0.75) leaves a centered ~1.0-unit label fully inside the axes, so
    the below placement cannot overflow into the y-axis ticks). Figure-wide
    11.44pt label size, 字号统一 never diverge per panel."""
    for i, (x, m, c) in enumerate(zip(xs, means, cIs)):
        dx = dxs[i] if dxs else 0
        dy = dys[i] if dys else 0
        if below and below[i]:
            ax.annotate(fr"${m:.3f} \pm {c:.3f}$", xy=(x, m - c),
                        xytext=(0, -5), textcoords="offset points",
                        va="top", ha="center", fontsize=11.44,
                        color=THEORY["dark"])
        else:
            ax.annotate(fr"${m:.3f} \pm {c:.3f}$", xy=(x, m + c),
                        xytext=(dx, 5 + dy), textcoords="offset points",
                        ha="center", fontsize=11.44, color=THEORY["dark"])


def main():
    # R30 zero-churn guard (orchestrator-authorized): an existing summary CSV
    # IS the data source — load it and never rewrite; only compute from
    # runs/ when the file is missing (analysis results are frozen).
    csv_path = Path("figures") / "rayleigh_monitor_ci_summary.csv"
    if csv_path.exists():
        sdf = pd.read_csv(csv_path)
        stats_ = {m: (float(mu), float(c))
                  for m, mu, c in zip(sdf.metric, sdf["mean"], sdf.ci95)}
        print(f"[guard] {csv_path} exists — loaded {len(stats_)} metrics, "
              "not rewritten")
    else:
        R = collect()
        print("=== multi-seed (mean ± 95% CI, n=5 CIFAR-ResNet seeds) ===")
        print(f"{'metric':28s} {'mean':>8s} {'±CI':>8s}")
        stats_ = {}
        rows = []
        for k, xs in R.items():
            mn, c, n = ci95(xs)
            stats_[k] = (mn, c)
            rows.append(dict(metric=k, n=n, mean=mn, ci95=c,
                             per_seed=";".join(f"{x:.3f}" for x in xs)))
            print(f"  {k:26s} {mn:8.3f} {c:8.3f}   per-seed "
                  f"{[round(x,3) for x in xs]}")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- figure: PDF, LaTeX mathtext ----
    # R30: true 2x2 equal-width grid. (a) whole-trajectory proxy comparison,
    # (b) monitor-vs-tune alignment, (c) EoS detrended correlation,
    # (d) EoS relative error — the former split (c) sub-panels promoted to
    # full panels (c)/(d). Focal estimator semantics: lambda_hat_grad is the
    # headline Rayleigh monitor in every existing panel ("dominates",
    # "monitor with g", "tracks lambda_max") -> damping-blue accent; the
    # competing estimators are comparators -> neutral slate (lambda_Dtheta,
    # Delta Theta) and theory_light gray (Tier-0 ||BN gamma||, seed-unstable).
    # Filled/open markers are the redundant non-hue channel (CB/grayscale).
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.2))
    fig.subplots_adjust(left=0.088, right=0.988, top=0.885, bottom=0.125,
                        wspace=0.34, hspace=0.58)

    # Panel (a): whole-trajectory detrended corr with CIs (proxy comparison)
    ax = axs[0, 0]
    # name / HVP-cost on separate tick levels: names ride the major ticks,
    # HVP costs are annotate'd below at a fixed pt offset (exact-gap trick,
    # see R28/R29 history; offsets track the rc tick size 7.0*SCALE).
    # R39 (fixplan §5.4/§7.3): rendered label unified with the paper's
    # \hat{\lambda}_g notation (was \hat{\lambda}_{\mathrm{grad}}).
    names = [r"$\hat{\lambda}_{g}$",
             r"$\hat{\lambda}_{\Delta\theta}$",
             r"$\|\mathrm{BN}\,\gamma\|$"]
    costs = ["(1 HVP)", "(1 HVP)", "(0 HVP)"]
    keys = ["lg_whole", "lu_whole", "bngamma_whole"]
    cols = [C["damping"], C["neutral"], C["theory_light"]]
    marks = ["o", "o", "s"]
    fills = ["full", "none", "none"]
    means = [stats_[k][0] for k in keys]
    cIs = [stats_[k][1] for k in keys]
    dot_whisker(ax, range(3), means, cIs, cols, marks, fills)
    # signed axis (detrended Pearson r in [-1,1]): the ||BN gamma|| CI dips
    # below zero (mean 0.247 ± 0.340 — one seed negatively correlated), so
    # keep a negative tick; the faint zero rule marks "no correlation"
    ax.axhline(0, color=THEORY["light"], lw=ST["theory"])
    ax.set_xticks(range(3))
    # (user round: the first point's value label is centered on x=0 and
    # poked past the old left spine (-0.45) into the y-tick labels; the
    # left edge now clears the ~0.46-data-unit half-width of the widest
    # label, and the right edge covers the last label's reach (2.46).)
    ax.set_xlim(-0.75, 2.6)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", which="major", pad=3.5)
    # per-label offsets calibrated to each name's ink descent (grad subscript
    # shallow, ||BN gamma|| norm bars deep) so the ink gap lands at ~4pt
    cost_offsets = [24.1, 23.4, 24.2]
    for xi, (cost, dy) in enumerate(zip(costs, cost_offsets)):
        ax.annotate(cost, xy=(xi, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -dy), textcoords="offset points",
                    va="top", ha="center", fontsize=11.44, color=THEORY["dark"])
    ax.set_ylabel(r"detrended Pearson corr" "\n" r"with $\lambda_{\max}$ (whole traj.)")
    # dynamic range (user round: hard-coded y-limits retired panel-wide);
    # 0 anchor keeps the no-correlation rule in view, 15%/21% pads below/
    # above the extreme whiskers clear the value annotations
    ax.set_ylim(*dyn_ylim(list(zip(means, cIs)), lo_pad=0.15, hi_pad=0.21,
                          anchor=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # first label sits BELOW its lower whisker tip, nudged 2pt right
    # (user round: centered above at x=0 collided with the y-axis ticks;
    # labels 2/3 stay above their tips, size unchanged figure-wide)
    annotate_vals(ax, range(3), means, cIs, below=[True, False, False])
    faint_ygrid(ax)
    panel_label(ax, "a", "Whole-trajectory proxy correlation")

    # Panel (b): g-vs-Dtheta alignment with CIs (the F20b monitor-vs-tune split)
    ax = axs[0, 1]
    means = [stats_["cos_g"][0], stats_["cos_dt"][0]]
    cIs = [stats_["cos_g"][1], stats_["cos_dt"][1]]
    dot_whisker(ax, [0, 1], means, cIs,
                [C["damping"], C["neutral"]], ["o", "o"], ["full", "none"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$\mathbf{g}$", r"$\Delta\theta$"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(r"mean $|\cos(\cdot, v_{\mathrm{top}})|$")
    # dynamic range (user round): the old fixed (0, 0.7) parked both dots
    # mid-panel with ~70% dead space below; the zoomed range follows the
    # whiskers with asymmetric pads (annotations ride above)
    ax.set_ylim(*dyn_ylim(list(zip(means, cIs)), lo_pad=0.5, hi_pad=0.95))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    annotate_vals(ax, [0, 1], means, cIs)
    faint_ygrid(ax)
    panel_label(ax, "b", "Monitor--tune alignment")

    # Panel (c): EoS-regime detrended correlation of lambda_hat_grad.
    # "tracks", not the historical "near-exact": panel (d) shows the ~17%
    # relative error — correlation-level agreement, not exactness.
    ax = axs[1, 0]
    eg_m, eg_c = stats_["lg_eos"]
    dot_whisker(ax, [0], [eg_m], [eg_c], [C["damping"]], ["o"], ["full"])
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([0])
    ax.set_xticklabels([r"EoS detrend $r$"])
    # dynamic range (user round): the detrended r may legitimately exceed 1
    # (tip ~1.011), so the range follows the whisker; pads clear the label
    ax.set_ylim(*dyn_ylim([(eg_m, eg_c)], lo_pad=0.45, hi_pad=0.95))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # (user round: (c)/(d) were the only panels without a y-label — added,
    # mirroring (a)'s phrasing so the 2x2 grid reads uniformly.)
    ax.set_ylabel(r"detrended Pearson corr" "\n" r"with $\lambda_{\max}$ (EoS)")
    annotate_vals(ax, [0], [eg_m], [eg_c])
    faint_ygrid(ax)
    panel_label(ax, "c", "EoS detrended correlation")

    # Panel (d): EoS-regime relative error of lambda_hat_grad (open marker:
    # same focal estimator, error quantity rather than agreement).
    ax = axs[1, 1]
    er_m, er_c = stats_["lg_err"]
    dot_whisker(ax, [0], [er_m], [er_c], [C["damping"]], ["o"], ["none"])
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([0])
    ax.set_xticklabels(["EoS rel. error"])
    ax.set_ylim(*dyn_ylim([(er_m, er_c)], lo_pad=0.45, hi_pad=0.95))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel(r"$\hat{\lambda}_{g}$ rel. error" "\n" r"(EoS regime)")
    annotate_vals(ax, [0], [er_m], [er_c])
    faint_ygrid(ax)
    panel_label(ax, "d", "EoS relative error")

    # No in-figure suptitle: the paper caption (fig:rayleigh in main.tex)
    # carries the summary message instead.
    out = Path("figures") / "fig1_rayleigh_monitor_ci.pdf"
    # keep the R29 pad: usetex tight-bbox under-measures the rotated (a)
    # y-label ink by ~1pt (re-verified R30: every edge keeps >=4pt margin)
    fig.savefig(out, pad_inches=0.06)
    plt.close(fig)
    print(f"\nsaved {out} (summary CSV: {'kept (guard)' if csv_path.exists() else 'written'})")


if __name__ == "__main__":
    main()
