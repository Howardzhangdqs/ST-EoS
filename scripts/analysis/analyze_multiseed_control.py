"""Paper A control-results aggregation (multi-seed CIs) — the predictor≠actuator
punchline figure. Everything built on a sharpness proxy locks back to 2/η; only
SAM (a global plant action) escapes. Vector PDF + LaTeX mathtext (figure_style).

Sources (per-seed CSVs from the parametrized experiments):
  F19: figures/eos_precond_summary_s{0..4}.csv  (MLP, 2/η=40)
  F12: figures/gapC_aggressive_retry_s{0..2}.csv (MLP, 2/η=40)
  F15: figures/gamma_controller_s{0..2}.csv      (CIFAR ResNet, momentum)

Usage: python scripts/analysis/analyze_multiseed_control.py
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

setup(scale=1.1678)   # R30: source canvas ~1.17x final size (report()-measured)
FIG = Path("figures")
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def ci(xs):
    xs = np.asarray([x for x in xs if x == x], float)
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    return float(xs.mean()), T.get(n, 1.96) * float(xs.std(ddof=1)) / sqrt(n)


def collect_f19():
    rows = []
    for s in range(5):
        p = FIG / f"eos_precond_summary_s{s}.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    df = pd.concat(rows)
    out = {}
    for opt in ["sgd", "eos_precond", "sam"]:
        out[opt] = ci(df[df.opt == opt]["lambda_max_final"])
    return out  # MLP, 2/η = 40


def collect_f12():
    rows = []
    for s in range(5):   # W-cap seeds 0-4 (s3/s4 CSVs added by the C1 rerun)
        p = FIG / f"gapC_aggressive_retry_s{s}.csv"
        if p.exists():
            rows.append(pd.read_csv(p).assign(seed=s))
    df = pd.concat(rows)
    return {"wcap": ci(df[df.variant == "wn_proj budget=1.01"]["lambda_max_final"])}


def collect_f15():
    """Returns (mean/CI dict, per-seed lists) — per-seed values kept so the
    γ-cap arm (removed from the figure, moved to a LaTeX table) can be
    printed at full precision for typesetting."""
    finals = {"baseline": [], "gcap": []}
    for s in range(3):
        p = FIG / f"gamma_controller_s{s}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        finals["baseline"].append(float(d[d.opt == "baseline"]["lambda_max"].iloc[-1]))
        finals["gcap"].append(float(d[d.opt == "gamma_cap"]["lambda_max"].iloc[-1]))
    return {k: ci(v) for k, v in finals.items()}, finals


def main():
    f19 = collect_f19(); f12 = collect_f12(); f15, f15_seed = collect_f15()
    print("=== Paper A control results (mean ± 95% CI) ===")
    # panel-(a) label convention (reviewer round 3): means at 1 dp; CI at
    # 1 dp normally, but when the half-width is < 0.05 print the ACTUAL CI
    # at 2 decimals (e.g. SAM 0.0417 → "± 0.04") — never a "±<0.05"
    # placeholder
    print("MLP (2/η=40) — panel-(a) labels as rendered:")
    for k, (m, c) in {**f19, **f12}.items():
        c_str = f"{c:.1f}" if c >= 0.05 else f"{c:.2f}"
        print(f"  {k:14s} λ_max_final = {m:6.1f} ± {c_str:>5s}   (actual CI = {c:.4f})")
    # γ-cap arm: REMOVED from the figure (two bars, n=3 — moving to a LaTeX
    # table per reviewer); the full-precision block below is the
    # typesetting source.
    print("CIFAR ResNet (momentum, 2/η=20 nominal) — γ-cap for LaTeX table:")
    for k in ("baseline", "gcap"):
        xs = f15_seed[k]
        m, c = f15[k]
        print(f"  {k:9s} n={len(xs)}  per-seed λ_max final: "
              + ", ".join(f"{x:.10g}" for x in xs))
        print(f"  {'':9s} mean = {m:.10g}   95% CI (t-dist) = ±{c:.10g}")

    # COMPRESSED (reviewer): single-panel figure — the two-bar γ-cap panel
    # (b) (76±22 vs 87±27, n=3) moved to a LaTeX table (values printed
    # above); panel (a) keeps its full content and roughly its slot aspect.
    # R30 style: arm semantics per script — SGD = neutral slate, W-cap =
    # aux olive, rank-1 plant = damping blue, SAM = teal (no sign-flipped
    # arm exists in this figure; threshold = theory gray).  Very faint
    # horizontal grid only (summary panel), frameless legend, short title.
    fig, ax = plt.subplots(figsize=(4.6, 2.9))

    # (a) MLP controllers vs SAM — the punchline (now the only panel)
    labels = [r"SGD", r"W-cap $(\|\mathbf{W}\|)$", r"rank-1 plant", r"SAM (global)"]
    means = [f19["sgd"][0], f12["wcap"][0], f19["eos_precond"][0], f19["sam"][0]]
    errs = [f19["sgd"][1], f12["wcap"][1], f19["eos_precond"][1], f19["sam"][1]]
    cols = [METHOD_COLORS["sgd"], METHOD_COLORS["wcap"],
            METHOD_COLORS["eos_precond"], METHOD_COLORS["sam"]]
    # OPAQUE bars (no alpha) + axisbelow: grid renders BEHIND the bars
    # (default grid zorder 1.5 > bar zorder 1 drew lines on top / showed
    # through the semi-transparent patches). Threshold axhline keeps its
    # Line2D zorder 2 above bars.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=THEORY["light"], alpha=0.45, lw=0.4)
    # ecolor: yerr whiskers default to the prop-cycle C0 (tab:blue) — pin
    # them to the neutral bar-edge gray so no legacy cycle color leaks in;
    # whisker weight from ST["errbar"] (R30 hierarchy).
    ax.bar(range(4), means, yerr=errs, capsize=ST["capsize"], color=cols,
           edgecolor="0.3", ecolor="0.3")
    for ln in ax.lines:  # yerr whisker + cap Line2D just created
        ln.set_color("0.3")                 # caps' marker face too
        ln.set_markerfacecolor("0.3")
        ln.set_linewidth(ST["errbar"])
    ax.axhline(40, color=THRESH_COLOR, ls="--", lw=ST["theory"],
               label=r"$2/\eta = 40$ (EoS threshold)")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\lambda_{\mathrm{raw}}$ final (power iter, ground truth)")
    # headroom for the bar-top value labels (top CI end 41.2 + 1.5-unit
    # offset + text ≈ 42.8); 0–50 keeps them clear of the title/upper spine
    ax.set_ylim(0, 50)
    ax.set_title(r"Proxy controls lock at $2/\eta$")
    ax.legend(loc="upper right")
    for i, (m, c) in enumerate(zip(means, errs)):
        # CI < 0.05 → actual value at 2 decimals (SAM: 0.0417 → "± 0.04"),
        # never the "±<0.05" placeholder (reviewer round 3)
        ci_lab = fr"\pm {c:.1f}" if c >= 0.05 else fr"\pm {c:.2f}"
        ax.text(i, m + c + 1.5, fr"${m:.1f} {ci_lab}$", ha="center", fontsize=9.34)

    # (b) CIFAR γ-cap: REMOVED (reviewer) — two bars at n=3 carry too little
    # information for a panel; exact values printed above for the LaTeX table.

    # No in-figure suptitle: the paper caption (fig:control in main.tex)
    # carries the summary message instead.
    out = savefig(fig, "26_multiseed_control_results")
    print(f"\nsaved {out}")
    print("figure panels remaining: 1 — MLP proxy controllers vs SAM "
          "(γ-cap panel removed → LaTeX table)")

    # CSV summary
    rows = []
    for regime, group, dct in [("MLP", "F19/F12", {**f19, "wcap": f12["wcap"]}),
                               ("CIFAR", "F15", f15)]:
        for k, (m, c) in dct.items():
            rows.append(dict(regime=regime, group=group, variant=k, lam_final_mean=m, lam_final_ci95=c))
    pd.DataFrame(rows).to_csv(FIG / "multiseed_control_summary.csv", index=False)
    print("saved figures/multiseed_control_summary.csv")


if __name__ == "__main__":
    main()
