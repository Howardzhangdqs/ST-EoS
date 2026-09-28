"""Rank-k interpolation aggregation — review round 2 headline experiment.

Late-window (step >= 2000) reference λ_max per arm across the rank-k family
(ARPACK-sensed damping k ∈ {1,2,5,10,20} and amplification α=1
k ∈ {5,20}), against the canonical SGD / rank-1 / SAM / amp1.0 bands.
Robust to partial data: arms with fewer runs than requested are reported
with their actual n; missing arms are skipped.

Outputs: printed table, figures/rankkx_interp.csv, and
figures/35_rankkx_interpolation.pdf (late λ vs k, both edit signs).

R30 aesthetic redesign (lane D2, structure redraw; spec: r29 diagnosis §4B
p19, 3.7 -> ~7.5): legend-driven -> data-driven. No central legend at all —
damping / amplification curves get line-end direct labels in a clean right
gutter; the current-minibatch-sensing control is a damping-family OPEN
marker (no cyan anywhere); SGD/SAM reference bands use a 6% tint over the
evaluated rank domain plus semantic center lines and right-column labels.
Uncertainty whiskers are visually subordinate to the mean trends. Semantic
palette / linestyle / marker conventions come from the global figure system.
Values, curves and arm semantics are unchanged; layout freeze remains lifted
for this figure (not a FROZEN_PAGES figure).

Data-change guard (R30): figures/rankkx_interp.csv is only written when
absent — a rerun never rewrites it, keeping the aggregate byte-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steos.analysis.figure_style import (
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    ST,
    setup,
)

LATE_STEP = 2000

ARMS = [
    # (display, run-prefix, family, k-value)  — k used for the x-axis
    ("SGD",           "cifar100_r18_sgd_detarpack_s",        None, None),
    ("SAM",           "cifar100_r18_sam_detarpack_s",        None, None),
    ("amp $\\alpha{=}1$ ($k{=}1$)", "cifar100_r18_amp1.0_detarpack_s",     "amp", 1),
    ("truelmax d1",   "cifar100_r18_truelmax_s",             "damp", 1),
    ("rankkx d1",     "cifar100_r18_rankkx_d1_s",            "damp", 1),
    ("rankkx d1 curb","cifar100_r18_rankkx_d1_curb_s",       "damp-curb", 1),
    ("rankkx d2",     "cifar100_r18_rankkx_d2_s",            "damp", 2),
    ("rankkx d5",     "cifar100_r18_rankkx_d5_s",            "damp", 5),
    ("rankkx d10",    "cifar100_r18_rankkx_d10_s",           "damp", 10),
    ("rankkx d20",    "cifar100_r18_rankkx_d20_s",           "damp", 20),
    ("rankkx a5",     "cifar100_r18_rankkx_a5_s",            "amp", 5),
    ("rankkx a20",    "cifar100_r18_rankkx_a20_s",           "amp", 20),
]


def ci95(xs):
    xs = np.asarray(xs, float)
    n = len(xs)
    if n < 2:
        return float(xs.mean()) if n else float("nan"), 0.0
    h = float(stats.t.ppf(0.975, n - 1) * xs.std(ddof=1) / np.sqrt(n))
    return float(xs.mean()), h


def late_stats(prefix: str) -> dict | None:
    """Exact-match '<prefix><seed>' dirs — a trailing '*' glob would also
    swallow sibling families (e.g. the canonical SGD prefix matching the
    'spec' runs), mixing protocols."""
    import re
    lams, fires, energies = [], [], []
    dirs = [d for d in Path("runs").iterdir()
            if d.is_dir() and re.fullmatch(re.escape(prefix) + r"\d+", d.name)]
    for d in sorted(dirs):
        f = d / "metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["step"] >= LATE_STEP]
        if df.empty or df["lambda_max"].isna().all():
            continue
        lams.append(float(df["lambda_max"].mean()))
        if "fired_frac" in df:
            fires.append(float(df["fired_frac"].iloc[-1]))
        # gradient-energy share captured by the edited subspace: from the
        # per-step jsonl, Σ_i cos²(g, v̂_i) — the per-step authority of the
        # rank-k edit (1.0 would mean the whole gradient lies in the subspace;
        # SAM acts along g itself, i.e. share 1 by construction).
        pj = d / "plant_per_step.jsonl"
        if pj.exists():
            import json as _json
            recs = [_json.loads(l) for l in open(pj)]
            recs = [r for r in recs if r["step"] >= LATE_STEP and "cos_g_v" in r]
            if recs:
                energies.append(float(np.mean(
                    [sum(c * c for c in r["cos_g_v"]) for r in recs])))
    if not lams:
        return None
    m, h = ci95(lams)
    return dict(n=len(lams), late_mean=m, late_ci=h, per_seed=lams,
                fired=float(np.mean(fires)) if fires else float("nan"),
                grad_energy_share=float(np.mean(energies)) if energies else float("nan"))


def main():
    setup(scale=0.9447)   # R31: 1/s, s=1.0584 (0.62\linewidth / 232.0pt)
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=1,
                    help="only report arms with at least this many complete runs")
    args = ap.parse_args()

    rows = []
    for name, pfx, fam, kk in ARMS:
        st = late_stats(pfx)
        if st is None or st["n"] < args.min_n:
            if st is None:
                print(f"[skip] {name}: no complete runs")
            continue
        rows.append(dict(arm=name, family=fam, k=kk, **{k: v for k, v in st.items()
                                                       if k != "per_seed"}))
        print(f"{name:16s} n={st['n']}  late λ = {st['late_mean']:6.2f} "
              f"± {st['late_ci']:4.2f}  fired={st['fired']:.3f}"
              f"  grad-energy-in-subspace={st['grad_energy_share']:.2e}")
    if not rows:
        raise SystemExit("no data yet")
    # R30 zero-data-change guard: only write the aggregate CSV when absent.
    csv_out = Path("figures/rankkx_interp.csv")
    if csv_out.exists():
        print(f"[guard] {csv_out} exists -- kept byte-identical, rewrite skipped")
    else:
        pd.DataFrame(rows).to_csv(csv_out, index=False)

    # ---- figure (R30 refinement: hierarchy + direct labels, no legend) -----
    # Canvas is intentionally unchanged: this figure is included at 0.62\linewidth.
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    fig.subplots_adjust(left=0.150, right=0.985, top=0.905, bottom=0.150)
    df = pd.DataFrame(rows)

    damp_color = METHOD_COLORS["damping"]
    amp_color = METHOD_COLORS["amplification"]
    sgd_color = METHOD_COLORS["sgd"]
    sam_color = METHOD_COLORS["sam"]
    axes_bg = plt.rcParams["axes.facecolor"]

    # Labels occupy a dedicated right gutter. Reference bands stop just after
    # the last tested rank, so the label column remains visually clean.
    label_x = 23.0
    band_xmin, band_xmax = 0.90, 21.4
    ref_line_xmax = 22.15
    lab_fs = plt.rcParams["ytick.labelsize"]

    # Main families: draw uncertainty first and semi-transparent, then overlay
    # the semantic mean trend. This keeps the large CIs visible without letting
    # them dominate the rank trend.
    family_specs = (
        ("damp", damp_color,
         METHOD_MARKERS["damping"], METHOD_LINESTYLES["damping"]),
        ("amp", amp_color,
         METHOD_MARKERS["amplification"], METHOD_LINESTYLES["amplification"]),
    )
    family_data = {}
    for fam, color, marker, ls in family_specs:
        sub = df[(df["family"] == fam) & df["k"].notna()].sort_values("k")
        family_data[fam] = sub
        if sub.empty:
            continue

        ax.errorbar(
            sub["k"], sub["late_mean"], yerr=sub["late_ci"],
            fmt="none", ecolor=color,
            elinewidth=ST["errbar"], capsize=ST["capsize"],
            alpha=0.52, zorder=2.2,
        )
        # note: both truelmax d1 and rankkx d1 remain at k=1, exactly as before
        ax.plot(
            sub["k"], sub["late_mean"],
            marker=marker, color=color, ls=ls,
            lw=ST["primary"], markersize=ST["marker"],
            markerfacecolor=color, markeredgecolor=color,
            zorder=3.1,
        )

    # Current-minibatch-sensing control: same damping hue, but an OPEN square.
    # Use the axes background as the face so the open marker stays legible even
    # where it overlaps the dense k=1 cluster; this is not a new semantic color.
    curb = df[df["family"] == "damp-curb"]
    if not curb.empty:
        cx = float(curb.iloc[0]["k"])
        cy = float(curb.iloc[0]["late_mean"])
        ch = float(curb.iloc[0]["late_ci"])
        ax.errorbar(
            [cx], [cy], yerr=[ch], fmt="none", ecolor=damp_color,
            elinewidth=ST["errbar"], capsize=ST["capsize"],
            alpha=0.52, zorder=2.4,
        )
        ax.plot(
            [cx], [cy], ls="none", marker="s",
            markersize=ST["marker"] * 1.30,
            markerfacecolor=axes_bg, markeredgecolor=damp_color,
            markeredgewidth=ST["theory"], zorder=4.2,
        )

    # SGD / SAM references: 6% CI tint only over the tested-rank domain, with
    # semantic line styles from figure_style (SGD solid, SAM dotted). The
    # center lines extend a little farther than the bands to lead into labels.
    references = (
        ("SGD", sgd_color, METHOD_LINESTYLES["sgd"]),
        ("SAM", sam_color, METHOD_LINESTYLES["sam"]),
    )
    for arm, color, ls in references:
        ref = df[df["arm"] == arm]
        if ref.empty:
            continue
        m = float(ref.iloc[0]["late_mean"])
        h = float(ref.iloc[0]["late_ci"])
        ax.fill_between(
            [band_xmin, band_xmax], [m - h, m - h], [m + h, m + h],
            color=color, alpha=0.06, linewidth=0, zorder=0.8,
        )
        ax.hlines(
            m, band_xmin, ref_line_xmax,
            colors=color, linestyles=ls, linewidth=ST["theory"],
            alpha=0.88, zorder=1.4,
        )
        ax.text(
            label_x, m, arm, color=color, fontsize=lab_fs,
            ha="left", va="center", zorder=4.5,
        )

    # Direct labels are tethered to the last available tested rank instead of
    # floating as unanchored text. Small vertical offsets keep damping clear of
    # the nearby SGD reference without changing any data position.
    damp = family_data.get("damp", pd.DataFrame())
    if not damp.empty:
        end = damp.iloc[-1]
        ax.annotate(
            "damping",
            xy=(float(end["k"]), float(end["late_mean"])), xycoords="data",
            xytext=(label_x, float(end["late_mean"]) + 0.65),
            textcoords="data", ha="left", va="center",
            fontsize=lab_fs, color=damp_color,
            arrowprops=dict(
                arrowstyle="-", color=damp_color,
                lw=ST["theory"] * 0.85, alpha=0.70,
                shrinkA=1.5, shrinkB=3.0,
            ),
            zorder=4.5,
        )

    amp = family_data.get("amp", pd.DataFrame())
    if not amp.empty:
        end = amp.iloc[-1]
        ax.annotate(
            "amplification\n" + r"$\alpha{=}1$",
            xy=(float(end["k"]), float(end["late_mean"])), xycoords="data",
            xytext=(label_x, float(end["late_mean"]) + 0.05),
            textcoords="data", ha="left", va="center",
            fontsize=lab_fs, color=amp_color, linespacing=1.35,
            arrowprops=dict(
                arrowstyle="-", color=amp_color,
                lw=ST["theory"] * 0.85, alpha=0.70,
                shrinkA=1.5, shrinkB=3.0,
            ),
            zorder=4.5,
        )

    # Compact leader for the open-square diagnostic; the larger open marker is
    # the primary identifier, while the annotation explains its role.
    if not curb.empty:
        ax.annotate(
            "current-minibatch\nsensing",
            xy=(cx, cy), xycoords="data",
            xytext=(0.055, 0.965), textcoords="axes fraction",
            ha="left", va="top", fontsize=lab_fs, color=damp_color,
            linespacing=1.35,
            arrowprops=dict(
                arrowstyle="-", lw=ST["theory"] * 0.75,
                color=damp_color, alpha=0.68,
                shrinkA=1.5, shrinkB=4.0,
            ),
            zorder=4.6,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.88, 62)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(20.5, 37)
    ax.set_yticks(range(22, 37, 2))
    ax.set_xlabel(r"edit rank $k$")
    ax.set_ylabel(r"late reference $\lambda_{\mathrm{ref}}$ "
                  r"(step $\geq$ 2000)")
    ax.set_title("Rank interpolation, ARPACK-sensed edits",
                 pad=4)

    out = Path("figures/35_rankkx_interpolation.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"=> {out} + figures/rankkx_interp.csv")


if __name__ == "__main__":
    main()
