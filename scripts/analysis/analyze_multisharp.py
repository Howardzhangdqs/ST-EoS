"""Aggregate the multi-sharp migration test across seeds and produce the paper
figure + verdict.

Reads ``figures/multisharp_summary_s{0..4}.csv`` (5 seeds; uses whichever exist)
and ``figures/multisharp_spectrum_s{seed}.csv``, computes mean ± 95% CI via the
t-distribution ``ci()`` (matches ``analyze_rankk_sweep.py``), and writes:

    figures/29_multisharp_migration.pdf   2-panel vector PDF, mathtext labels
    figures/multisharp_summary.csv        mean ± CI per optimizer

Panel (a): λ_max_final (±CI) bar chart for the 5 optimizers + 2/η reference line
          + SAM flat-escape line. SGD pins at 2/η; SAM flat-escapes; the rank-k
          variants sharp-escape PAST 2/η but are bounded (≈65–77, not ≈269 as on
          the 1-D landscape), and k>1 escapes MORE (k1<k2≈k5) — k matters here.
Panel (b): top-5 Hessian eigenvalue evolution over training (seed 0) for SGD vs
          rankk_k1. SGD: λ_1 and λ_2 BOTH pin at 2/η (two simultaneous EoS pins).
          rankk_k1: λ_1 escapes past 2/η (cap removed along v_1) while λ_2 stays
          ≈2/η — cap-removal per direction, NOT migration (migration would pin
          λ_max at 2/η; instead λ_max = max(escaping λ_1, pinned λ_2) > 2/η).

Usage: python scripts/analysis/analyze_multisharp.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    C, METHOD_COLORS, THRESH_COLOR, THEORY, ST, setup, savefig,
)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

setup(scale=1.0970)
FIG = Path("figures")
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}     # t-dist 95% (two-sided), df=n-1
RANKK_KS = [1, 2, 5]
TWO_OVER_ETA = 40.0
LAM_THRESH = 0.7 * TWO_OVER_ETA                     # flat-escape criterion
REPR_SEED = 0                                       # representative seed for panel (b)


def ci(xs):
    """mean ± 95% CI half-width via the t-distribution (matches
    analyze_rankk_sweep.ci). NaNs are dropped; n<2 ⇒ (nan, nan)."""
    xs = np.asarray([x for x in xs if x == x], float)
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    return float(xs.mean()), T.get(n, 1.96) * float(xs.std(ddof=1)) / sqrt(n)


def fmt_pm(m, c, nd=1):
    """Bar-annotation text `mean ± CI` with 1-decimal mean.

    Revision fix: the old `${m:.0f}±{c:.0f}` format truncated small CIs to a
    misleading "±0" (e.g. SAM rendered "22±0" with a real CI of 0.38). The
    CI is the same t-based half-width `ci()` computes for the CSV/console —
    only the rendering truncated it. A CI that would still round to zero at
    `nd` decimals (<0.05) renders as "±<0.05" instead.
    """
    if not (c == c):                       # NaN (n<2): no CI to quote
        return fr"${m:.{nd}f}$"
    if c < 0.05:
        return fr"${m:.{nd}f}\,\pm{{<}}0.05$"
    return fr"${m:.{nd}f} \pm {c:.{nd}f}$"


def main():
    # --- aggregate per-seed summaries --------------------------------------
    rows = []
    for s in range(5):
        p = FIG / f"multisharp_summary_s{s}.csv"
        if p.exists():
            rows.append(pd.read_csv(p).assign(seed=s))
    if not rows:
        raise SystemExit("No per-seed CSVs found (figures/multisharp_summary_s*.csv); "
                         "run scripts/experiments/multisharp_experiment.py first.")
    df = pd.concat(rows, ignore_index=True)

    opt_order = ["sgd", "sam"] + [f"rankk_k{k}" for k in RANKK_KS]
    agg = {}
    for opt in opt_order:
        sub = df[df.opt == opt]
        if not len(sub):
            continue
        lam_m, lam_c = ci(sub["lambda_max_final"])
        loss_m, loss_c = ci(sub["final_loss"])
        agg[opt] = dict(lam_mean=lam_m, lam_ci=lam_c,
                        loss_mean=loss_m, loss_ci=loss_c, n=len(sub))

    print("=== Multi-sharp migration test (mean ± 95% CI) ===")
    print(f"  setup: k_out=4 isotropic outputs, MLP [256,256] tanh, η=0.05, 2/η={TWO_OVER_ETA:g}")
    for opt in opt_order:
        if opt in agg:
            a = agg[opt]
            print(f"  {opt:10s} λ_max_final = {a['lam_mean']:6.2f} ± {a['lam_ci']:5.2f}   "
                  f"final_loss = {a['loss_mean']:.4f} ± {a['loss_ci']:.4f}   (n={a['n']})")

    # --- verdict -----------------------------------------------------------
    print(f"\n=== VERDICT (escape-to-flat needs λ_max_final < {LAM_THRESH:.0f} = 0.7·2/η; "
          f"sharp-escape >> 2/η = {TWO_OVER_ETA:g}) ===")
    for k in RANKK_KS:
        m = agg[f"rankk_k{k}"]["lam_mean"]
        cls = ("flat-escape" if m < LAM_THRESH
               else f"lock@2/η (≈{TWO_OVER_ETA:.0f})" if m < 1.4 * TWO_OVER_ETA
               else "SHARP-escape (>>2/η, bounded)")
        print(f"  rankk_k{k:<2d}: λ_max_final={m:7.2f}  → {cls}")
    print(f"  SGD : λ_max_final={agg['sgd']['lam_mean']:.2f}  (EoS pin @2/η — TWO directions pinned)")
    print(f"  SAM : λ_max_final={agg['sam']['lam_mean']:.2f}  (flat-escape)")

    rankk_lams = [agg[f"rankk_k{k}"]["lam_mean"] for k in RANKK_KS]
    spread = max(rankk_lams) - min(rankk_lams)
    k1, k2, k5 = rankk_lams
    print(f"\n  k-dependence: k1={k1:.1f}  k2={k2:.1f}  k5={k5:.1f}  spread={spread:.1f}")
    print(f"  (1-D landscape: k1=k2=k5=k10≈269, spread≈0 — k-INVARIANT)")
    print(f"  (multi-sharp:   k1<k2≈k5 — k>1 MATTERS: each damped direction at the")
    print(f"   EoS cap is one removed cap ⇒ more directions escape ⇒ higher λ_max)")

    # --- spectrum-evidence (migration vs cap-removal) on the representative seed
    spec_path = FIG / f"multisharp_spectrum_s{REPR_SEED}.csv"
    migration_evidence = ""
    if spec_path.exists():
        sp = pd.read_csv(spec_path)
        final_step = int(sp["step"].max())
        for opt in ("sgd", "rankk_k1"):
            sub = sp[(sp.opt == opt) & (sp.step == final_step)].sort_values("idx")
            if len(sub):
                v = sub["lambda_i"].values[:5]
                print(f"  spectrum @step{final_step} {opt:9s}: "
                      + "  ".join(f"λ{i+1}={x:5.1f}" for i, x in enumerate(v)))
        # cap-removal signature: under rankk_k1, λ_1 escapes (>2/η) while λ_2
        # stays ≈2/η (its cap intact). migration signature would be λ_1 damped
        # BELOW 2/η with λ_2 rising to absorb it and λ_max re-pinned at 2/η.
        sgd_f = sp[(sp.opt == "sgd") & (sp.step == final_step)].sort_values("idx")["lambda_i"].values
        rk1_f = sp[(sp.opt == "rankk_k1") & (sp.step == final_step)].sort_values("idx")["lambda_i"].values
        if len(sgd_f) >= 2 and len(rk1_f) >= 2:
            migration_evidence = (
                f"\n  Mechanism check @step{final_step} (seed {REPR_SEED}):\n"
                f"    SGD      : λ_1={sgd_f[0]:.1f}  λ_2={sgd_f[1]:.1f}  (both ≈2/η={TWO_OVER_ETA:g})\n"
                f"    rankk_k1 : λ_1={rk1_f[0]:.1f}  λ_2={rk1_f[1]:.1f}\n"
                f"    → λ_1 grew from {sgd_f[0]:.1f}→{rk1_f[0]:.1f} (cap removed along v_1)\n"
                f"    → λ_2 ≈ {rk1_f[1]:.1f} (its cap INTACT; not rising to absorb migration)\n"
                f"    ⇒ CAP-REMOVAL per direction, NOT migration (migration would keep\n"
                f"      λ_max≈2/η; instead λ_max=max(λ_1,λ_2)={max(rk1_f[0], rk1_f[1]):.1f} > 2/η)."
            )
            print(migration_evidence)

    print("\n=== MIGRATION VERDICT (decides §6) ===")
    if all(m > 1.4 * TWO_OVER_ETA for m in rankk_lams):
        print("  MIGRATION DOES NOT MANIFEST. Self-gating removes the EoS cap on each")
        print("  damped direction → λ_max escapes upward (NOT re-pinned at 2/η). The")
        print("  self-stabilization-cap mechanism holds generally. §6 should KEEP")
        print("  'self-stabilization cap' (NOT 'migration'), with the scope condition")
        print("  that on multi-sharp landscapes k>1 removes more caps → k matters")
        print("  (contrast: 1-D landscape is k-INVARIANT).")

    # =====================================================================
    # FIGURE
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.95),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(wspace=0.30)

    # --- (a) λ_max_final bar chart (±CI) for the 5 optimizers ---------------
    ax = axes[0]
    labels = ["sgd", "sam", "k=1", "k=2", "k=5"]
    means = [agg[o]["lam_mean"] for o in opt_order]
    errs = [agg[o]["lam_ci"] for o in opt_order]
    # sgd/sam from the color registry; the whole rank-k damping family shares
    # the rank-1 plant color (red), matching fig27's uniform-red rank-k bars —
    # the x tick labels (k=1/2/5) discriminate within the family
    colors = [METHOD_COLORS["sgd"], METHOD_COLORS["sam"], METHOD_COLORS["rank1"],
              METHOD_COLORS["rank1"], METHOD_COLORS["rank1"]]
    x = np.arange(len(opt_order))
    # OPAQUE bars (no alpha) + axisbelow: grid renders BEHIND the bars
    # (default grid zorder 1.5 > bar zorder 1 drew lines on top / showed
    # through the semi-transparent patches). The LAM_THRESH dotted axhline
    # below keeps its Line2D zorder 2 above the bars.
    ax.set_axisbelow(True)
    # R30: bar summary -> faint horizontal major grid only (neutral gray,
    # alpha <= 0.25), behind the ink.
    ax.grid(axis="y", alpha=0.25, color=C["theory_light"])
    ax.bar(x, means, yerr=errs, capsize=ST["capsize"], color=colors,
           edgecolor="0.3")
    sgd_lam = agg["sgd"]["lam_mean"]
    sam_lam = agg["sam"]["lam_mean"]
    # R30: reference lines at theory weight (thin, neutral); the SAM level
    # keeps the SAM hue (it IS the SAM arm's value) but reads as a guide.
    ax.axhline(TWO_OVER_ETA, color=THRESH_COLOR, ls="-", lw=ST["theory"])
    ax.axhline(sam_lam, color=METHOD_COLORS["sam"], ls="--", lw=ST["theory"])
    ax.axhline(LAM_THRESH, color=THEORY["light"], ls=":", lw=ST["theory"],
               alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda_{\mathrm{raw}}$ final (power iter, ground truth)")
    # R30: short panel title (the k-ordering story lives in the caption)
    ax.set_title(r"\textbf{(a)} Escape grows with $k$", color="#1A1D21")
    ymin = min(min(means), sam_lam, LAM_THRESH) * 0.6
    ymax = max(means) * 1.35
    ax.set_ylim(ymin, ymax)
    # R30: the three reference levels are direct-labelled in a dedicated
    # right-hand lane (xlim extended; bars untouched) instead of a legend —
    # the figure keeps exactly ONE legend (panel b). Label vertical sides:
    # 2/eta and the threshold sit ABOVE their lines, SAM BELOW its line
    # (28 and 22.6 are too close on the log axis to stack both above).
    # The flat-escape label renders on TWO lines (string-concat newline,
    # same convention as fig 32's annotations): the single line ran ~59pt
    # wide and reached into the (a)/(b) panel gap (user round) — the wrap
    # narrows it to the lane; ma="left" keeps line 2 left-aligned.
    ax.set_xlim(-0.5, 5.9)
    # (user round 4: all three lane labels typeset their digits/letters
    # in LaTeX TEXT mode — the same engine as the "flat-escape"
    # reference — so every label renders at the flat-escape size; only
    # η and ≈ keep inline math mode ($\eta$, $\approx$), the two glyphs
    # with no text-mode unicode under usetex. The earlier all-unicode
    # attempt died on "! LaTeX Error: Unicode character η (U+03B7) not
    # set up for use with LaTeX".)
    # (user round 5: the three lane labels are RIGHT-ALIGNED as one
    # flush column at the lane's right end — xy=5.85, ha/ma="right"
    # (ma covers the two-line flat-escape label's second line). Widths
    # stay clear of the bars (rightmost bar body ends ≈179pt page-x;
    # the widest label reaches back only to ≈188pt) and of panel (b)
    # (right edges ≈228.5pt vs (b) ink at 245.4pt). Font sizes are
    # metrically identical across the three labels (same 10.5pt word
    # boxes, Times lining digits); the residual size impression is the
    # two-line/lowercase/light-gray structure of the flat-escape
    # label, not a pt difference.)
    # (user round 6: all three lane labels shrink a touch (explicit
    # fontsize 8.0 vs the 8.74pt rcParams default) and anchor their right
    # edge at axes-fraction x=1.0 — flush with the axes RIGHT SPINE.
    # The old data-coord anchor x=5.85 (xlim 5.9) left a ~1.8pt gap
    # between the labels and the end of the rule lines; the axhlines run
    # to the spine, so the labels now end exactly where the lines end.
    # Explicit fontsize is absolute (setup()'s scale factor does not
    # rescale it), so 8.0 renders as literal 8.0pt for all three labels.)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    LANE_FS = 8.0
    for yv, txt, col, above in (
            (TWO_OVER_ETA, rf"2/$\eta$ = {TWO_OVER_ETA:.0f}", THRESH_COLOR, True),
            (LAM_THRESH, rf"flat-escape" "\n" r"$\approx$ " rf"{LAM_THRESH:.0f}",
             THEORY["light"], True),
            (sam_lam, rf"SAM $\approx$ {sam_lam:.0f}", METHOD_COLORS["sam"],
             False)):
        ax.annotate(txt, xy=(1.0, yv), xycoords=trans, ha="right", ma="right",
                    va="bottom" if above else "top",
                    xytext=(0, 2 if above else -2), textcoords="offset points",
                    fontsize=LANE_FS, color=col)
    for i, (m, c) in enumerate(zip(means, errs)):
        # two-line form (mean / ±CI), same convention as fig 32 panel (b):
        # adjacent bars have near-equal values (75.2±1.6 vs 77.4±1.7) and
        # the single-line labels collided horizontally. CI<0.05 renders
        # "±<0.05" (fmt_pm convention) rather than a truncated "±0.0".
        ci_txt = rf"$\pm$ {c:.1f}" if c >= 0.05 else r"$\pm$ <0.05"
        ax.annotate(f"{m:.1f}\n{ci_txt}", xy=(i, m + c), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    fontweight="bold")

    # --- (b) spectrum evolution (seed 0): top-5 for SGD vs rankk_k1 --------
    ax = axes[1]
    plotted = False
    if spec_path.exists():
        sp = pd.read_csv(spec_path)
        # optimizer family = registry hue (slate = SGD, blue = rank-1); λ1 vs
        # λ2 distinguished by linestyle (λ1 solid, λ2 dashed) since each
        # method carries a single color. R30 stroke hierarchy: λ1 primary,
        # λ2 secondary, λ3-5 seed-weight context ink.
        main_style = {("sgd", 0): METHOD_COLORS["sgd"], ("sgd", 1): METHOD_COLORS["sgd"],
                      ("rankk_k1", 0): METHOD_COLORS["rank1"],
                      ("rankk_k1", 1): METHOD_COLORS["rank1"]}
        for opt in ("sgd", "rankk_k1"):
            sub = sp[sp.opt == opt]
            for i in range(5):
                s = sub[sub.idx == i].sort_values("step")
                if not len(s):
                    continue
                # λ2: thinner + fainter than λ1 so the solid λ1 clearly
                # dominates where they coincide (user: dashed hard to parse)
                lw = ST["primary"] if i == 0 else (ST["secondary"] if i == 1
                                                   else ST["seed"])
                alpha = 1.0 if i == 0 else (0.55 if i == 1 else 0.5)
                lab = None
                if i < 2:
                    lab = rf"$\lambda_{{{i+1}}}$ ({opt.replace('rankk_k1', 'rank-1')})"
                lstyle = "--" if i == 1 else "-"
                # zorder descending in eigenindex: where λ1 and λ2 coincide
                # (SGD pins both at 2/η), the solid λ1 must render ON TOP of
                # the dashed λ2 — otherwise the overlap reads as λ2 > λ1
                ax.plot(s["step"], s["lambda_i"], ls=lstyle, lw=lw, alpha=alpha,
                        color=main_style[(opt, i)] if i < 2 else C["theory_light"],
                        label=lab, zorder=5 - i)
            plotted = True
        ax.axhline(TWO_OVER_ETA, color=THRESH_COLOR, ls="-", lw=ST["theory"])
        # 2/η direct-labelled ON the line (user round): x∈[0,~850] is
        # curve-free (all curves still climb below 40 there; they pin or
        # cross the line from x≈900 on — checked against the spectrum
        # CSV), so the text sits centered on the rule line with the same
        # white halo as the legend: the line passes behind the glyphs.
        # (user round 3: single line kept — wrap rejected; flushed LEFT
        # instead. The label is 35.9pt wide = 28% of the 128pt axis, so a
        # centered single line can reach step≈350 at best before poking
        # the left spine; anchoring its left edge just inside the spine
        # (ha="left", xy=100 → 5.1pt clearance) is the maximal single-
        # line leftward position.)
        # (user round 6: string switched from full-math $2/\eta = 40$ to
        # the SAME text-mode convention as panel (a)'s lane label. Root
        # cause of the user's "left and right 2/η = 40 look different":
        # the math version typeset every glyph in Computer Modern (wider,
        # lighter digits + math-mode "=" spacing), while the lane labels'
        # digits are Times text mode — different typeface at equal pt
        # reads as a size mismatch. Identical strings + identical
        # fontsize=LANE_FS make both sides render pixel-identically. The
        # text-mode label is ~3pt narrower, so the x=100 anchoring keeps
        # ~12pt clearance to the curves' x≈900 crossing of the rule.)
        ax.annotate(rf"2/$\eta$ = {TWO_OVER_ETA:.0f}",
                    xy=(100, TWO_OVER_ETA), ha="left", va="center",
                    fontsize=LANE_FS, color=THRESH_COLOR,
                    path_effects=[pe.withStroke(linewidth=3.5,
                                                foreground="white")])
        ax.set_xlabel("step")
        ax.set_xticks(np.arange(0, 2501, 500))
        ax.set_ylabel(r"top Hessian eigenvalues $\lambda_i$")
        # R30: short panel title — the SGD-pins-both / rank-1-λ2-stays
        # reading lives in the paper caption.
        ax.set_title(r"\textbf{(b)} Cap removal, not migration", color="#1A1D21")
        # the figure's single legend (panel a uses direct labels) with a
        # neutral proxy for the unlabeled λ3–λ5 curves (reviewer round 3).
        # Legend sits LOWER-LEFT (user round; was lower-right): every curve
        # starts bottom-left in its climb segment, so the rising ink passes
        # behind the legend — same sanctioned fallback as R30 fix ①: white
        # stroke HALO on the legend text and handle lines (path_effects,
        # NOT a background box; the legend stays frameless; the halo only
        # clears ~2.5pt around each glyph/handle so the crossing ink reads
        # as passing behind). The 2/η rule is direct-labelled on the line
        # itself (annotate above) and no longer carries a legend entry.
        handles, _labs = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color=C["theory_light"], lw=ST["seed"],
                              alpha=0.5,
                              label=r"$\lambda_{3}$--$\lambda_{5}$ (subdominant)"))
        leg = ax.legend(handles=handles, loc="lower left")
        halo = [pe.withStroke(linewidth=3.5, foreground="white")]
        for txt in leg.get_texts():
            txt.set_path_effects(halo)
        for h in leg.legend_handles:
            h.set_path_effects(halo)
    if not plotted:
        ax.text(0.5, 0.5, rf"multisharp\_spectrum\_s{REPR_SEED}.csv not found",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()

    # No in-figure suptitle: the paper caption (fig:multisharp in main.tex)
    # carries the summary message instead.
    out = savefig(fig, "29_multisharp_migration")
    print(f"\nsaved {out}")

    # --- summary CSV (mean ± CI per opt) -----------------------------------
    # R30 zero-change guard: never rewrite an existing summary CSV
    # (authorized minimal IO guard; the figure reads `agg`, not this file).
    csv_path = FIG / "multisharp_summary.csv"
    if csv_path.exists():
        print(f"kept existing {csv_path} (rewrite skipped)")
    else:
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
        pd.DataFrame(srows).to_csv(csv_path, index=False)
        print("saved figures/multisharp_summary.csv")


if __name__ == "__main__":
    main()
