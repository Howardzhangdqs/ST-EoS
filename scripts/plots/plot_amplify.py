"""Figure 33 — the rank-1 AMPLIFICATION (sign-flip) control on the MLP
full-batch testbed (reviewer-driven control of Corollary cor:amplify).

Aggregates figures/amplify_{summary,traj}_s{0..4}.csv (from
scripts/experiments/amplify_rank1_experiment.py, the rankk_sweep protocol) into:
  (a) seed-mean λ_max trajectories, log y (the damping arm sharp-escapes to
      ~262; the amplifying arms settle below 2/η=40) with the 2/η line and
      the predicted equilibria λ*=2/((1+α)η) as thin dashed neutral-gray
      reference levels;
  (b) late-window (step≥1500) observed (filled) vs predicted (open) dots on
      a horizontal log-x dot plot — rows are methods, the 2/η=40 threshold
      is a thin vertical line. Replaces the r29 broken-axis bar chart.

Layout (R30 redesign lane D2; 2026-09-24 fig-review rounds 1-3):
  * panel (a): curves fill the full panel width (no in-axes label gutter);
    a compact frameless TWO-COLUMN legend in the empty lower-right corner
    identifies all six strokes (col 1 = reference methods, col 2 = the α
    ramp — 3 rows only, so the block never climbs into the curves);
  * semantic palette: SGD slate, SAM teal, damping blue; the amplifying
    family keeps ONE rose hue with LIGHTNESS encoding α (.25 lightest →
    1 deepest) — monotone shade for monotone α; linestyle
    (solid/dashed/dotted) stays as the grayscale-safe redundant channel
    (the old opacity-only separation was unreadable);
  * predictions are all neutral theory-light gray, thin dashed — reference
    levels never borrow a method hue;
  * panel (b): rows = methods (short α row labels — rose ramp + caption
    carry family membership), observed = filled dot (method hue) + 95% CI
    whisker, predicted = open dot on the same row, one frameless in-panel
    key.

Data-change guard (R30): figures/amplify_summary_agg.csv is only computed
and written when absent — a rerun loads the existing aggregate and never
rewrites it (this script historically regenerated the CSV alongside the
figure; the guard pins it byte-identical).

CLI (2026-09-24, fig.9 canonical rebuild): --traj takes an explicit list of
per-seed traj CSVs (order = seed order; per-seed summaries are the sibling
amplify_summary_* files in the same directory), --agg-out/--fig-out redirect
the outputs to explicit paths. NO glob/mtime auto-selection anywhere; the
no-argument default is the fixed figures/ filename loop. --manifest is an
optional informational provenance note only (skipped with a note if the
path is absent — this experiment writes no manifest).
"""
from __future__ import annotations

import argparse
import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    C,
    ST,
    THRESH_COLOR,
    panel_label,
    setup,
    savefig,
)

FIG = Path("figures")
TWO_OVER_ETA = 40.0
T975 = 2.776  # t(0.975, df=4)
LATE = 1500
ARMS = ["sgd", "sam", "rankk_k1", "amp_a0.25", "amp_a0.5", "amp_a1"]
AMP_ALPHAS = ["amp_a0.25", "amp_a0.5", "amp_a1"]
# R30 semantic palette: reference arms keep their method hues; the
# amplifying family keeps its ONE rose hue (aesthetic_diagnosis §4B p8) —
# (user round, 2026-09-24 fig-review: one flat rose + linestyle-only
# separation was unreadable where the three mid-run trajectories
# interleave, and the 0.95/0.80/0.65 opacity steps were too subtle to
# parse. Lightness now encodes α WITHIN the family — weakest arm lightest,
# strongest arm at full stroke — so the arms stay one hue family AND are
# separable at a glance; linestyle (solid/dashed/dotted) remains the
# grayscale-safe redundant channel. The opacity hack is dropped: full-
# opacity ramp shades track cleanly.)
def _mix_white(hex_color: str, frac: float) -> str:
    """Blend a hex color toward white by frac (0 = unchanged, 1 = white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    up = lambda c: round(c + (255 - c) * frac)
    return "#{:02x}{:02x}{:02x}".format(up(r), up(g), up(b))


AMP_SHADE = {"amp_a0.25": _mix_white(C["amplification"], 0.38),
             "amp_a0.5": _mix_white(C["amplification"], 0.16),
             "amp_a1": C["amplification"]}
COLOR = {"sgd": C["neutral"], "sam": C["sam"], "rankk_k1": C["damping"],
         **AMP_SHADE}
# redundant non-color coding WITHIN the α family — α=0.25 solid /
# α=0.5 dashed / α=1 dotted (monotone in α). Reference arms are all SOLID
# (user round 5: sam + damping join sgd on solid strokes — the three hues
# separate them; the figure-level legend below the panels carries the
# mapping).
LSTYLE = {"sgd": "-", "sam": "-", "rankk_k1": "-",
          "amp_a0.25": "-", "amp_a0.5": "--", "amp_a1": ":"}
# right-gutter direct-label text — superseded by the in-axes legend
# (user round 3); COL1/COL2 labels retired, single LABEL map feeds both
# the panel-(a) legend and the panel-(b) row labels.
LABEL = {"sgd": "SGD", "sam": "SAM", "rankk_k1": "damping",
         "amp_a0.25": r"$\alpha{=}.25$",
         "amp_a0.5": r"$\alpha{=}.5$",
         "amp_a1": r"$\alpha{=}1$"}
ROW_LABEL = {"sgd": "SGD", "sam": "SAM", "rankk_k1": "damping",
             "amp_a0.25": r"$\alpha{=}.25$",
             "amp_a0.5": r"$\alpha{=}.5$",
             "amp_a1": r"$\alpha{=}1$"}


def predicted(opt: str) -> float | None:
    if opt.startswith("amp_a"):
        return TWO_OVER_ETA / (1.0 + float(opt.split("_a")[1]))
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="With no --traj, falls back to the canonical fixed-name loop "
               "over figures/amplify_{{traj,summary}}_s{0..4}.csv (explicit "
               "names, NOT mtime/glob selection).")
    ap.add_argument("--traj", nargs="+", metavar="CSV", default=None,
                    help="explicit per-seed traj CSVs (order = seed order); "
                         "each one's per-seed summary is the sibling "
                         "amplify_summary_* file in the same directory "
                         "(required for the 'final' column; hard error if "
                         "missing — never fabricated)")
    ap.add_argument("--manifest", default=None,
                    help="optional manifest.json for a provenance NOTE only "
                         "(informational; never used to fabricate inputs). "
                         "If the path does not exist, a note is printed and "
                         "the manifest is skipped")
    ap.add_argument("--agg-out", default=None,
                    help="aggregate CSV output path (default: "
                         "figures/amplify_summary_agg.csv)")
    ap.add_argument("--fig-out", default=None,
                    help="figure output path (default: "
                         "figures/33_rank1_amplify.pdf)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # oversized source canvas shrunk at the 0.75\linewidth include — scale
    # keeps final on-paper strokes/fonts at the R30 baseline (see report()).
    # (user round 2026-09-24 #1: canvas 5.6→6.3in + wspace 0.37→0.50 opened
    # the (a)/(b) gutter ~30% on paper while the scale compensation
    # (×6.3/5.6) kept every on-paper font/stroke at its old size.
    # user round #2: canvas 6.3→7.0in + wspace 0.50→0.62 — the gutter now
    # hosts TWO label columns (panel (a)'s direct labels at its left edge,
    # panel (b)'s row labels at its right edge; the amp rows dropped the
    # redundant "amp" prefix — the rose shades + caption carry the family),
    # so panel (a) no longer surrenders its right quarter to the old
    # in-axes label gutter (xlim 4150 vs data end 2900); curves fill the
    # full panel width. Scale compensation ×7.0/5.6 keeps on-paper sizes.
    # round #5: 7.0→7.6in — the legend moves OUT of the panels onto its own
    # bottom band (fig.legend below), so both panels reclaim the legend's
    # corner real estate; height scales with width to hold the on-paper
    # figure height constant at the 0.75\linewidth include.
    # round #6: the include widened 0.75→0.95\linewidth (user: the figure
    # read small — a quarter of the line width sat idle). The ×0.75/0.95
    # factor compensates the bigger paper footprint so on-paper fonts and
    # strokes keep the R30 calibration while the PANELS grow ~27%.)
    setup(scale=1.4883 * 7.6 / 5.6 * 0.75 / 0.95 * 8.5 / 9.23 * 8.5 / 8.62)  # ×8.5/9.23 R38 tag-size unification (2-pass bbox-tight trim)

    if args.manifest:
        m = Path(args.manifest)
        if m.exists():
            n_rows = sum(1 for _ in m.open())
            print(f"[manifest] {m} exists ({n_rows} lines) — noted, "
                  "informational only")
        else:
            print(f"[manifest] {m} not found — skipping (this experiment "
                  "writes no manifest; nothing fabricated)")

    if args.traj:
        traj_files = [Path(p) for p in args.traj]
        for p in traj_files:
            if not p.is_file():
                raise SystemExit(f"missing traj CSV: {p}")
        summ_files = [p.parent / p.name.replace("traj", "summary")
                      for p in traj_files]
        for p, q in zip(traj_files, summ_files):
            if p == q or not q.is_file():
                raise SystemExit(
                    f"cannot derive per-seed summary for {p} (expected "
                    f"sibling {q}) — refusing to fabricate")
    else:
        traj_files = [FIG / f"amplify_traj_s{s}.csv" for s in range(5)]
        summ_files = [FIG / f"amplify_summary_s{s}.csv" for s in range(5)]

    trajs, summaries = [], []
    for s, (tp, sp) in enumerate(zip(traj_files, summ_files)):
        t = pd.read_csv(tp)
        t["seed"] = s
        trajs.append(t)
        sm = pd.read_csv(sp)
        sm["seed"] = s
        summaries.append(sm)
    traj = pd.concat(trajs, ignore_index=True)
    summ = pd.concat(summaries, ignore_index=True)

    agg_out = (Path(args.agg_out) if args.agg_out
               else FIG / "amplify_summary_agg.csv")
    # R30 zero-data-change guard: reuse an existing aggregate verbatim.
    if agg_out.exists():
        agg = pd.read_csv(agg_out, index_col=0)
        print(f"[guard] {agg_out} exists — loaded, rewrite skipped "
              "(aggregate kept byte-identical)")
        print(agg.to_string(float_format=lambda x: f"{x:7.3f}"))
    else:
        rows = []
        for opt in ARMS:
            late = traj[(traj.opt == opt) & (traj.step >= LATE)]
            for s in range(5):
                ls = late[late.seed == s]
                rows.append(dict(opt=opt, seed=s,
                                 late_mean=float(ls.lambda_max.mean()),
                                 frac_below=float(
                                     (ls.lambda_max < TWO_OVER_ETA).mean()),
                                 final=float(summ[(summ.opt == opt) &
                                                  (summ.seed == s)]
                                             .lambda_max_final.iloc[0])))
        per = pd.DataFrame(rows)
        agg = per.groupby("opt").agg(
            late_mean=("late_mean", "mean"),
            late_ci=("late_mean", lambda x: T975 * x.std(ddof=1) / sqrt(5)),
            frac_below=("frac_below", "mean"), final=("final", "mean"))
        agg["predicted"] = [predicted(o) for o in agg.index]
        agg_out.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_out)
        print(agg.to_string(float_format=lambda x: f"{x:7.3f}"))

    fig = plt.figure(figsize=(7.6, 3.44))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.070,
                          right=0.988, top=0.862, bottom=0.26, wspace=0.48)

    # ---------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0, 0])
    lab_fs = plt.rcParams["ytick.labelsize"]
    X_END = 2960.0          # rules span the full panel — no in-axes label gutter
    # predicted equilibria: thin dashed neutral-gray reference levels
    for opt in AMP_ALPHAS:
        ax.hlines(predicted(opt), -70, X_END, color=C["theory_light"],
                  ls="--", lw=ST["theory"], zorder=1.5)
    # seed-mean trajectories (labeled — the lower-right legend identifies
    # all six strokes)
    for opt in ARMS:
        g = traj[traj.opt == opt].groupby("step").lambda_max.mean()
        ax.plot(g.index, g.to_numpy(), color=COLOR[opt], ls=LSTYLE[opt],
                lw=ST["primary"], zorder=3, label=LABEL[opt])
    # 2/η threshold (reference, dark neutral — never a method hue)
    ax.hlines(TWO_OVER_ETA, -70, X_END, color=THRESH_COLOR, ls="-",
              lw=ST["theory"] * 1.15, zorder=2)
    ax.text(X_END - 120, TWO_OVER_ETA * 1.075, "$2/\\eta{=}40$", ha="right",
            va="bottom", fontsize=lab_fs, color=THRESH_COLOR)

    # figure-level legend BELOW the panels (user round 5: pulled out of the
    # axes entirely — the in-axes lower-right block kept grazing the α=1
    # late arm, and both panels now own their full area). ONE row of six;
    # font at 0.88× tick size keeps the row inside the canvas while the
    # on-paper size (~7.6pt) stays legend-grade. Order = [SGD, SAM,
    # damping | α=.25, α=.5, α=1].)
    ax.set_yscale("log")
    ax.set_ylim(2, 420)
    ax.set_yticks([10, 100])
    ax.set_xlim(-70, X_END)
    handles, labels = ax.get_legend_handles_labels()
    # (cross-figure key spacing: bottom-key gap ~6 pt at 0.95\linewidth in
    # the paper, matching 39_long_recipe / 43_bsharp_arms — see R38 audit.)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.529, 0.0233), ncol=6, frameon=False,
               fontsize=0.88 * lab_fs, handlelength=1.3,
               handletextpad=0.4, columnspacing=0.8)

    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_{\mathrm{raw}}$")
    panel_label(ax, "a", "Sign-flip control")

    # ---------------------------------------------------------------- (b)
    # horizontal dot plot: methods on y, λraw on log x. Observed = filled
    # dot + 95% CI whisker; predicted = open dot on the same row.
    axb = fig.add_subplot(gs[0, 1])
    axb.grid(True, axis="x", which="major", alpha=0.35, lw=0.4,
             color=C["theory_light"])
    dy_obs, dy_pred = -0.14, 0.14
    for i, opt in enumerate(ARMS):
        m, ci = agg.loc[opt, "late_mean"], agg.loc[opt, "late_ci"]
        axb.errorbar([m], [i + dy_obs], xerr=[ci], fmt="o", color=COLOR[opt],
                     lw=0, elinewidth=ST["errbar"], capsize=ST["capsize"],
                     markersize=ST["marker"] * 1.1, markerfacecolor=COLOR[opt],
                     markeredgecolor=COLOR[opt], zorder=3.2)
        p = predicted(opt)
        if p is not None:
            axb.plot([p], [i + dy_pred], "o", markersize=ST["marker"] * 1.1,
                     markerfacecolor="none", markeredgecolor=COLOR[opt],
                     mew=1.1 * ST["errbar"], zorder=3.2)
    axb.axvline(TWO_OVER_ETA, color=THRESH_COLOR, ls="-",
                lw=ST["theory"] * 1.15, zorder=2)
    # label sits RIGHT of the rule (user: the old centered spot put the
    # vertical line through the glyph, cutting "2/η" in two); 1.15x on the
    # log axis ≈ 9 pt clearance at panel width
    axb.text(TWO_OVER_ETA * 1.15, -1.02, "$2/\\eta$", ha="left", va="top",
             fontsize=lab_fs, color=THRESH_COLOR)

    axb.set_xscale("log")
    axb.set_xlim(13.5, 330)
    axb.set_xticks([20, 40, 100, 200])
    axb.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axb.minorticks_off()
    axb.set_ylim(5.5, -1.25)
    axb.set_yticks(range(len(ARMS)))
    axb.set_yticklabels([ROW_LABEL[o] for o in ARMS])
    axb.set_xlabel(r"$\lambda_{\mathrm{raw}}$ (late mean)")
    panel_label(axb, "b", "Late equilibria vs.\\ prediction")

    # one-line frameless key for the filled/open pairing (the figure's
    # only legend)
    key_obs = Line2D([], [], marker="o", ls="none", color=C["neutral"],
                     markersize=ST["marker"] * 1.1)
    key_pred = Line2D([], [], marker="o", ls="none", color=C["theory_light"],
                      markerfacecolor="none", markersize=ST["marker"] * 1.1)
    axb.legend([key_obs, key_pred], ["observed", "predicted"],
               loc="lower right", frameon=False, handletextpad=0.3,
               borderaxespad=0.4, labelspacing=0.35)

    if args.fig_out:
        out = Path(args.fig_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # same save parameters as figure_style.savefig (bbox tight via
        # rcParams + pad_inches=0.02), but to an explicit full path
        fig.savefig(out, pad_inches=0.02)
        plt.close(fig)
    else:
        out = savefig(fig, "33_rank1_amplify")
    print(f"=> {out} + {agg_out}")


if __name__ == "__main__":
    main()
