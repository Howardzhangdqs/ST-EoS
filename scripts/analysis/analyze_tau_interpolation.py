"""τ-interpolation: two-panel dose-response (reference λmax + batch sharpness).

Panel (a) — late-window reference λmax for the τ-interpolation family,
recomputed under ONE estimator for all four τ values:

    per-seed late-window mean of metrics.csv:lambda_max (step ≥ LATE_STEP)
    → cross-seed mean ± 95% CI (Student-t, ppf 0.975, ddof=1)

Panel (b) — the stochastic-metric companion (review ask: close the loop
between the mechanism blend and the quantity the stochastic EoS theory
regulates): same estimator on the batch sharpness λ_step =
E_B[g_BᵀH_Bg_B/‖g_B‖²] = metrics.csv:lhg_bs_mean (m=4 minibatches,
logged every 200 steps).

Arms (endpoint identities per scripts/analysis/validate_tau_match.py, which verified
sam_interp τ=1.0 ≡ `sam` and τ=0.0 ≡ `linsam_bnmatched` bit-exactly on the
deterministic protocol):

    τ=0.00  cifar100_r18_linsam_bnmatched_mb_s{0..4}
    τ=0.25  cifar100_r18_sam_interp_t0.25_mb_s{0..4}
    τ=0.50  cifar100_r18_sam_interp_t0.5_mb_s{0..4}
    τ=1.00  cifar100_r18_sam_mb_s{0..4}          (plain SAM endpoint arm)

No paper numbers are hardcoded anywhere in this script; it only reads raw
runs. Diagnostics additionally report (a) the step ≥ 3000 window and
(b) the lam1_b124 column at step ≥ 3000 with a normal-approx CI
(1.96·sem) — this combination is the provenance of the published
interior-point values (reverse-engineered, see audit report), and is
kept here so the CSV documents exactly what each convention yields.

Outputs:
    figures/tau_interpolation_summary.csv   (+ bsharp_* columns)
    figures/46_tau_interpolation.pdf   (top: late reference λmax(τ);
                                        bottom: batch sharpness λ_step(τ);
                                        mean±95% CI, thin guide lines, no fits)
Usage:
    .venv/bin/python scripts/analysis/analyze_tau_interpolation.py [--plan {wide,narrow}]

Round-29 typography plans (--plan): the paper includes this figure at a
fraction of \\linewidth (396pt text block); final on-page font size =
source size * scale with scale = 396*frac/W (W = 3.2in*72 = 230.4pt
exactly — the tight trim is disabled so W is deterministic).  Both plans
below hold LAYOUT-ONLY constants (font sizes / line widths / panel-title
wording / canvas margins); the data, CI, and CSV paths are untouched.
There is deliberately NO in-panel "annotation" role: the protocol details
(step >= 2000, fixed-512 ARPACK, m=4 minibatches) moved to the paper
caption, leaving three text roles (panel title / axis label / tick
label), each sized so its FINAL size hits the round-29 QA bands
(tick 7.2±0.3 / axis 7.8±0.3 / title 8.0±0.3).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt          # noqa: E402
from steos.analysis.figure_style import (  # noqa: E402
    C, THEORY, setup, savefig,
)

LATE_STEP = 2000        # paper-canonical late window (analyze_bsharp_arms.py)
LATE_STEP_ALT = 3000    # diagnostic window for the mixing hypothesis
VALUE_COL = "lambda_max"     # canonical fixed-512 reference ARPACK λmax
BSHARP_COL = "lhg_bs_mean"   # batch sharpness λ_step (m=4 minibatches)
PROV_COL = "lam1_b124"       # cross-batch-study column (provenance diag)

TAU_ARMS = [
    (0.00, "cifar100_r18_linsam_bnmatched_mb_s"),
    (0.25, "cifar100_r18_sam_interp_t0.25_mb_s"),
    (0.50, "cifar100_r18_sam_interp_t0.5_mb_s"),
    (1.00, "cifar100_r18_sam_mb_s"),
]

# ---------------------------------------------------------------------------
# Side-by-side layout (2026-09-25, user request: panels left/right, not
# stacked). Single typography plan — the old vertical-era "wide"/"narrow"
# plans (0.50/0.30\linewidth, 3.2x4.3in canvas) are retired.
# Rendered from main.tex at 396pt*frac; final on-page size = source*scale
# with scale = 396*frac/W, W = 5.4in*72 = 388.8pt exactly (deterministic:
# the tight trim is disabled below, see the savefig.bbox note in main()).
#   side : frac 0.62 -> scale 0.6315; sources tick 12.67 / axis 12.67 /
#          title 13.46 give finals 8.00 / 8.00 / 8.50 pt — the same finals
#          the vertical wide+narrow plans converged to (zero size change).
# Roles: title = left-aligned panel title; axis = x/y axis labels;
#        tick = tick labels.  No annotation role (protocol -> caption).
# Line widths back-computed at the same scale: errorbar 1.43 (final 0.90),
# guide 1.16 (final 0.73), spine 1.27 (final 0.80), grid 0.71 (final 0.45),
# marker 6.13 (final 3.87), markeredgewidth 1.22 (final 0.77), capsize
# 4.09 (final 2.58).
# Margins/wspace reserve room for the usetext quads (title 6pt pad +
# ~1.2*fs above each axes; rotated ylabel + tick stack left of EACH panel,
# xlabel + tick stack below) so every glyph stays >=2pt inside the
# untrimmed canvas.
# ---------------------------------------------------------------------------
PLANS = {
    "side": dict(   # 0.62\linewidth, panels (a) left / (b) right
        figsize=(5.4, 2.55), wspace=0.34,
        left=0.115, right=0.985, top=0.86, bottom=0.20,
        tick=12.67, axis=12.67, title=13.46,
        lw_spine=1.27, lw_grid=0.71, lw_errorbar=1.43, lw_guide=1.16,
        marker_size=6.13, marker_edgewidth=1.22, capsize=4.09,
    ),
}


def discover_seeds(prefix: str) -> list[int]:
    """Exact-match runs/<prefix><seed>/ with a metrics.csv."""
    out = []
    for d in (ROOT / "runs").iterdir():
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", d.name)
        if m and (d / "metrics.csv").exists():
            out.append(int(m.group(1)))
    return sorted(out)


def ci95(xs: list[float]) -> tuple[float, float]:
    """Cross-seed Student-t 95% CI half-width (repo convention)."""
    a = np.asarray(xs, float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), 0.0
    h = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return float(a.mean()), h


def protocol_of(run: str) -> dict:
    mf = ROOT / "runs" / run / "manifest.json"
    m = json.loads(mf.read_text()) if mf.exists() else {}
    return {k: m.get(k) for k in ("steps", "eval_batch", "outcome_every",
                                  "rho", "opt", "tau")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="tau-interpolation two-panel dose-response figure")
    ap.add_argument(
        "--plan", choices=sorted(PLANS), default="side",
        help="typography plan: 'side' sizes fonts for the 0.62\\linewidth "
             "side-by-side render (default). Layout only — data, CI, and "
             "CSV output are identical.")
    args = ap.parse_args()
    plan: dict = PLANS[args.plan]
    setup()
    # Round-29 layout guard (NOT part of the locked font chain): the shared
    # savefig() wrapper trims with bbox="tight" + pad 0.02in, but matplotlib
    # under-measures usetex glyph quads by ~1-2pt, which clipped the panel
    # title / rotated ylabel at the page edge.  Disable the trim for this
    # figure only: W is then exactly figsize*72pt (deterministic scale), and
    # the per-plan margins above keep every glyph >=2pt inside the canvas.
    plt.rcParams["savefig.bbox"] = None
    # spine weight is a pure layout knob (not part of the locked font chain)
    plt.rcParams["axes.linewidth"] = plan["lw_spine"]
    rows = []
    print(f"{'τ':>5} {'n':>2}  {'late(t≥2000)':>16}  {'diag(t≥3000)':>16}  "
          f"{'prov(b124,t≥3000,1.96sem)':>26}  {'bsharp(t≥2000)':>16}")
    for tau, prefix in TAU_ARMS:
        seeds = discover_seeds(prefix)
        late, alt, prov, bsharp, protos = [], [], [], [], set()
        for s in seeds:
            run = f"{prefix}{s}"
            df = pd.read_csv(ROOT / "runs" / run / "metrics.csv")
            late.append(float(df.loc[df["step"] >= LATE_STEP, VALUE_COL].mean()))
            alt.append(float(df.loc[df["step"] >= LATE_STEP_ALT, VALUE_COL].mean()))
            if PROV_COL in df.columns:
                prov.append(float(df.loc[df["step"] >= LATE_STEP_ALT, PROV_COL].mean()))
            if BSHARP_COL in df.columns:
                bsharp.append(float(
                    df.loc[df["step"] >= LATE_STEP, BSHARP_COL].mean()))
            p = protocol_of(run)
            protos.add((p["steps"], p["eval_batch"], p["outcome_every"], p["rho"]))
        lm, lh = ci95(late)
        am, ah = ci95(alt)
        bm, bh = ci95(bsharp)
        if prov:
            pm = float(np.mean(prov))
            ph = float(1.96 * np.std(prov, ddof=1) / np.sqrt(len(prov)))
        else:
            pm, ph = float("nan"), float("nan")
        rows.append(dict(tau=tau, n=len(seeds), late_mean=lm, late_ci95=lh,
                         late_mean_t3000=am, late_ci95_t3000=ah,
                         late_mean_b124_t3000=pm, late_ci95_b124_t3000_norm=ph,
                         bsharp_mean=bm, bsharp_ci95=bh,
                         bsharp_n=len(bsharp)))
        print(f"{tau:>5.2f} {len(seeds):>2}  {lm:>7.2f} ±{lh:<7.2f}  "
              f"{am:>7.2f} ±{ah:<7.2f}  {pm:>12.2f} ±{ph:<12.2f}  "
              f"{bm:>7.2f} ±{bh:<7.2f}")
        if len(protos) > 1:
            print(f"      WARN protocol drift within arm: {protos}")

    # R30 zero-change guard: if the summary CSV already exists (shipped with
    # the paper), read it and never rewrite — authorized minimal IO guard,
    # no analysis logic change. Compute + write only when missing.
    csv_path = ROOT / "figures" / "tau_interpolation_summary.csv"
    if csv_path.exists():
        out = pd.read_csv(csv_path)
        print(f"[guard] using existing {csv_path} (rewrite skipped)")
    else:
        out = pd.DataFrame(rows)
        out.to_csv(csv_path, index=False, float_format="%.4f")

    # ---- two-panel figure, side by side: (a) reference λmax(τ) LEFT,
    # (b) batch sharpness λ_step(τ) RIGHT
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=plan["figsize"], sharex=True,
        gridspec_kw=dict(width_ratios=[1, 1], wspace=plan["wspace"]))
    # R30 stroke/color-only pass: the whole tau-interpolation family is the
    # SAM/perturbed-point dose-response -> teal; the two panels (reference
    # vs batch sharpness metric) keep their existing o/s marker channel.
    # Grid: point summary -> faint HORIZONTAL major grid only.
    for ax, mcol, hcol, color, marker, ylabel in (
            (axa, "late_mean", "late_ci95", C["sam"], "o",
             r"late reference $\lambda_{\mathrm{ref}}$"),
            (axb, "bsharp_mean", "bsharp_ci95", C["sam"], "s",
             r"batch sharpness $\lambda_{\mathrm{step}}$")):
        ax.errorbar(out.tau, out[mcol], yerr=out[hcol], marker=marker,
                    ms=plan["marker_size"], lw=0, elinewidth=plan["lw_errorbar"],
                    capsize=plan["capsize"], markeredgewidth=plan["marker_edgewidth"],
                    color=color, zorder=3)
        ax.plot(out.tau, out[mcol], lw=plan["lw_guide"], color=color,
                alpha=0.5, zorder=2)  # visual guide only — NOT a fit
        ax.set_ylabel(ylabel, fontsize=plan["axis"])
        ax.margins(x=0.05)
        ax.grid(axis="y", alpha=0.25, color=THEORY["light"],
                lw=plan["lw_grid"])
        ax.tick_params(labelsize=plan["tick"])
    # Round-29: short panel titles (protocol details live in the paper
    # caption); rendered as left-aligned titles ABOVE each axes so they
    # cannot collide with the tau=0 error bar at the data maximum.
    axa.set_title(r"\textbf{(a)} Fixed-batch reference",
                  fontsize=plan["title"], color="#1A1D21")
    axb.set_title(r"\textbf{(b)} Batch sharpness",
                  fontsize=plan["title"], color="#1A1D21")
    # side-by-side: each column has its own bottom x-axis -> τ label and
    # ticks on BOTH panels (sharex keeps the tick locations consistent)
    for ax in (axa, axb):
        ax.set_xlabel(r"$\tau$", fontsize=plan["axis"])
        ax.set_xticks([0.0, 0.25, 0.5, 1.0])
    fig.subplots_adjust(left=plan["left"], right=plan["right"], top=plan["top"],
                        bottom=plan["bottom"])
    savefig(fig, "46_tau_interpolation")
    csv_state = "kept existing" if csv_path.exists() else "wrote"
    print(f"\n{csv_state} figures/tau_interpolation_summary.csv "
          f"({len(out)} rows) and wrote figures/46_tau_interpolation.pdf "
          f"[--plan {args.plan}: tick {plan['tick']} / axis {plan['axis']} / "
          f"title {plan['title']} pt source]")


if __name__ == "__main__":
    main()
