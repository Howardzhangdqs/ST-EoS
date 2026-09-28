"""Batch sharpness ACROSS stochastic arms — review round 5 (reviewer priority 1).

Round 3 (Finding 43a) measured the EoSS batch sharpness
E_B[g_BᵀH_Bg_B/‖g_B‖²] on the SGD arm only.  This script answers the
round-5 reviewer's acceptance-critical ask: measure it on EVERY central arm
— SGD (mb2), SAM, linSAM (n=5), rank-1 amplification (α=0.5), rank-1
damping (rankkx d1) — and ask the question that closes the stochastic-EoS
loop:

  does an arm that lowers the FIXED-REFERENCE λmax (SAM) also move the
  quantity the stochastic EoS theory actually regulates (the batch
  sharpness), or does the batch sharpness stay pinned at 2/η while only the
  spectral geometry changes?

Outputs (late window step ≥ 2000, mean ± 95% CI over seeds; paired Δ vs SGD
on shared seeds):
  figures/bsharp_arms_summary.csv   (R30 data-layer guard: written only
                                     when MISSING — never rewritten)
  figures/43_bsharp_arms.pdf        R30 restyle on the FROZEN footprint
                                     535.975 x 249.735 pt (savefig_fixed):
                                     (a) reference λmax per arm; (b) batch
                                     sharpness per arm — per-seed traces as
                                     faint background texture, seed means
                                     bold in the semantic method colors,
                                     one shared frameless key; (c) forest
                                     plot of the late-window paired Δλstep
                                     vs SGD (methods on the y axis, zero
                                     as a thin theory-gray rule)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from steos.analysis.figure_style import (
    C, ST, FROZEN_PAGES, INCLUDE_FRAC, TEXTWIDTH_PT,
    setup, savefig_fixed, frozen_figsize, panel_label,
)
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

LATE_STEP = 2000
ETA, MU = 0.1, 0.9
GD = 2 / ETA                      # 20 — full-batch GD / momentum-free reference
HB = 2 * (1 + MU) / ETA           # 38 — heavy-ball large-batch plateau

# R30 semantic palette (figure_style.py): SGD neutral slate, rank-1 damping
# blue, rank-1 amplification rose, linSAM purple, SAM teal.
# R39 grayscale-redundancy lane (fixplan_r15 §6): every arm now carries a
# DISTINCT linestyle + marker shape on the bold mean curves so the five-arm
# panels stay parseable in grayscale print (identity still rides on hue;
# the per-arm markers in panel (c) were already the established channel).
# Per-seed traces inherit the arm linestyle (no marker, faint texture).
ARMS = [
    ("SGD",          "cifar100_r18_sgd_detarpack_mb2_s", C["neutral"],  "-",          "o"),
    ("rank-1 damp",  "cifar100_r18_rankkx_d1_mb_s",      C["damping"],  "--",         "s"),
    ("rank-1 amp",   "cifar100_r18_amp05_mb_s",          C["amplification"], "-.",    "^"),
    ("linSAM",       "cifar100_r18_linsam_mb_s",         C["linsam"],   ":",          "D"),
    ("SAM",          "cifar100_r18_sam_mb_s",            C["sam"],      (0, (5, 2)),  "v"),
]

# source-canvas -> final-physical-size magnification for the FROZEN page
# (report("figures/43_bsharp_arms.pdf", "43_bsharp_arms") prints it)
FIG43_SCALE = (FROZEN_PAGES["43_bsharp_arms"][0]
               / (INCLUDE_FRAC["43_bsharp_arms"] * TEXTWIDTH_PT))  # ≈1.6507


def ci95(xs):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) < 2:
        return (float(xs.mean()) if len(xs) else float("nan")), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def smooth32(steps, mean):
    """Fig-32 parity display smoothing (user round: apply the established
    optimization set to fig 43): Gaussian filter on log10 of the bold
    cross-seed mean, with the kernel width expressed in STEPS so the degree
    matches figs 30/32 at any sampling density — σ = 12 samples on the
    every-20-step grid ≡ a 240-step kernel — plus the same init-transient
    guard: the first 3 samples stay raw, crossfaded log-linearly into the
    smoothed curve over samples 2–12. DISPLAY-LAYER ONLY: every statistic
    (panel (c) forest, summary CSV, console) uses the raw per-seed values.
    """
    steps = np.asarray(steps, float)
    dt = float(np.median(np.diff(steps))) if len(steps) > 2 else 200.0
    # kernel width in STEPS is the invariant (240 steps, the fig-30/32
    # degree) → σ in SAMPLES = 240/dt. On this grid (dt≈200) that is a light
    # 1.2-sample Gaussian; the earlier 12×dt/20 formula produced σ=120
    # samples on the 26-point array and collapsed the curve to its global
    # mean.
    sigma = max(240.0 / dt, 0.6)
    lg = np.log10(mean)
    sm_lg = gaussian_filter1d(lg, sigma=sigma, mode="reflect")
    w = np.clip((np.arange(len(lg)) - 2) / (12 - 2), 0, 1)
    return 10 ** ((1 - w) * lg + w * sm_lg)


def paired_ci(d, level=0.90):
    """Paired mean difference ± t-CI (ddof=1).  Default 0.90 keeps the
    historical stdout/CSV/TOST-adjacent behavior; the R27 ordinary paired
    difference renders at 0.95 (see panel (c))."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), 0.0, float(d.mean()), float(d.mean())
    m = float(d.mean())
    h = float(stats.t.ppf(0.5 + level / 2, len(d) - 1)
              * d.std(ddof=1) / np.sqrt(len(d)))
    return m, h, m - h, m + h


def tost_p(d, margin):
    """Two one-sided tests p-value for |mean(d)| < margin (paired, t)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), False
    se = d.std(ddof=1) / np.sqrt(n)
    if se <= 0:
        return 0.0 if abs(d.mean()) < margin else 1.0, abs(d.mean()) < margin
    df = n - 1
    t_lo = (d.mean() - (-margin)) / se     # H0: mean <= -margin
    t_hi = (d.mean() - margin) / se        # H0: mean >= +margin
    p = max(stats.t.cdf(t_lo, df), 1 - stats.t.cdf(t_hi, df))
    return float(p), bool(p < 0.05)


def load_arm(prefix):
    out = {}
    for d in sorted(Path("runs").glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            df = pd.read_csv(d / "metrics.csv")
            if "lhg_bs_mean" in df.columns:
                out[int(s)] = df
            else:
                print(f"[skip] {d.name}: no lhg_bs_mean column")
    return out


def main():
    # R30: typography/strokes from the shared visual system; the R29 local
    # font overrides are gone (setup(scale) carries them).
    setup(scale=FIG43_SCALE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--late-step", type=int, default=LATE_STEP)
    args = ap.parse_args()

    data = {}
    for label, prefix, _, _, _ in ARMS:
        runs = load_arm(prefix)
        if runs:
            data[label] = runs
        else:
            print(f"[arm missing] {label} ({prefix}*) — no runs yet")
    if not data:
        raise SystemExit("no armed runs found")

    rows, traj = [], {}
    for label, _, color, _, _ in ARMS:
        if label not in data:
            continue
        seeds = sorted(data[label])
        late_bs, late_lmax, late_curmb, pooled = [], [], [], []
        for s in seeds:
            df = data[label][s]
            m = df[df["step"] >= args.late_step]
            late_bs.append(float(m["lhg_bs_mean"].mean()))
            late_lmax.append(float(m["lambda_max"].mean()))
            pooled.append(m["lhg_bs_mean"].to_numpy())
            if "lhg_curmb" in m.columns:
                late_curmb.append(float(m["lhg_curmb"].mean()))
        traj[label] = (seeds, late_bs, late_lmax, color)
        bs_m, bs_h = ci95(late_bs)
        lm_m, lm_h = ci95(late_lmax)
        pooled = np.concatenate(pooled)
        pooled = pooled[~np.isnan(pooled)]
        frac_above_gd = float((pooled > GD).mean())
        frac_above_hb = float((pooled > HB).mean())
        # Taylor re-query content (sam/linsam arms only; NaN elsewhere)
        tres = [float(x) for x in
                (data[label][s][data[label][s]["step"] >= args.late_step]
                 ["taylor_res_rel"].dropna().tolist()
                 if "taylor_res_rel" in data[label][s].columns else []) for s in seeds]
        # one-sample t-test of the per-seed late batch sharpness vs 2/η
        if len(late_bs) >= 2 and np.std(late_bs, ddof=1) > 0:
            tstat, pvs = stats.ttest_1samp(late_bs, GD)
        else:
            pvs = float("nan")
        rows.append(dict(arm=label, n_seeds=len(seeds),
                         lmax_ref_mean=lm_m, lmax_ref_ci95=lm_h,
                         bsharp_mean=bs_m, bsharp_ci95=bs_h,
                         bsharp_frac_above_2overeta=frac_above_gd,
                         bsharp_frac_above_hb=frac_above_hb,
                         taylor_res_rel_mean=(float(np.mean(tres))
                                              if tres else float("nan")),
                         bsharp_p_vs_2overeta=float(pvs),
                         bsharp_curmb_mean=(float(np.mean(late_curmb))
                                            if late_curmb else float("nan"))))
        pv = f" p(vs 2/η)={pvs:.3f}" if not np.isnan(pvs) else ""
        print(f"{label:14s} n={len(seeds)}  ref λmax {lm_m:6.2f} ± {lm_h:4.2f}   "
              f"batch sharpness {bs_m:6.2f} ± {bs_h:4.2f}   frac>2/η "
              f"{frac_above_gd:.2f}{pv}")

    # paired contrasts vs SGD on shared seeds (+ TOST at the pre-specified
    # margin 0.25·|SGD−SAM| on the same quantity — three-tier language)
    if "SGD" in data:
        for label in data:
            if label == "SGD":
                continue
            shared = sorted(set(data[label]) & set(data["SGD"]))
            if len(shared) < 2:
                continue
            for q, vals in (("bsharp", 1), ("lmax_ref", 2)):
                d = np.array([traj[label][vals][traj[label][0].index(s)]
                              for s in shared]) - \
                    np.array([traj["SGD"][vals][traj["SGD"][0].index(s)]
                              for s in shared])
                m, h, lo, hi = paired_ci(d)
                if "SAM" in traj:
                    sam_m = float(np.mean(traj["SAM"][vals]))
                    sgd_m = float(np.mean(traj["SGD"][vals]))
                    margin = 0.25 * abs(sgd_m - sam_m)
                    tp, tok = tost_p(d, margin)
                else:
                    margin, tp, tok = float("nan"), float("nan"), False
                rows.append(dict(arm=f"Δ({label}−SGD) {q}", n_seeds=len(shared),
                                 lmax_ref_mean=m, lmax_ref_ci95=h,
                                 bsharp_mean=float("nan"), bsharp_ci95=float("nan"),
                                 bsharp_frac_above_2overeta=float("nan"),
                                 bsharp_p_vs_2overeta=float("nan"),
                                 bsharp_curmb_mean=float("nan"),
                                 tost_margin=margin, tost_p=tp, tost_pass=tok))
                eq = (" SIGNIFICANT difference (CI excludes 0)" if lo > 0 or hi < 0
                      else (" TOST-equivalent" if tok
                            else " (no measurable difference at this power)"))
                print(f"paired Δ({label}−SGD) {q:8s} = {m:+.2f}  90% CI "
                      f"[{lo:+.2f},{hi:+.2f}]  margin ±{margin:.2f}{eq}"
                      f"  (n={len(shared)})")

    out = pd.DataFrame(rows)
    # R30 data-layer guard (orchestrator-authorized minimal IO guard): the
    # published summary CSV is never rewritten; it is written only when
    # missing. The in-memory table above is still computed for the prints.
    out_path = Path("figures/bsharp_arms_summary.csv")
    if out_path.exists():
        print(f"[guard] {out_path} exists — kept untouched")
    else:
        out.to_csv(out_path, index=False)
        print(f"=> {out_path}")

    # ---- figure: (a) λ_ref(t), (b) λ_step(t) faint seeds + bold mean,
    #      (c) late-window paired-Δ forest plot --------------------------
    # FROZEN footprint 535.975 x 249.735 pt via savefig_fixed (no tight
    # crop); margins are hand-set so nothing touches the page border.
    fig, axes = plt.subplots(
        1, 3, figsize=frozen_figsize("43_bsharp_arms"),
        # (user round: 面板间隙偏大 — wspace 0.56→0.28 widens all three
        # panels so they fill the frozen width; the (b) ylabel and
        # (c) tick labels still clear the gaps at this setting.)
        gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.28))
    # (user round: legend moved OUT of panel (a) onto a one-row band below
    # all three panels — fig-33 pattern. bottom 0.165→0.27 makes room for
    # ticks + xlabel + the legend row inside the frozen footprint. R38
    # audit: 0.27→0.291 then 0.291→0.2575 tunes the key gap to ~6 pt at
    # 0.95\linewidth, matching 33/39 — the key itself can't move (its
    # anchor already sits at the canvas edge, savefig_fixed would clip),
    # so the panels do the moving.)
    fig.subplots_adjust(left=0.068, right=0.993, top=0.90, bottom=0.2575)
    # (user round: ab 间隔近一些，bc 远一些 — (b) shifts left toward (a),
    # (c) shifts right toward the canvas edge; panel sizes unchanged. (c)'s
    # shift is capped at +0.003 so its right spine stays 2pt inside the
    # frozen canvas — +0.008 let the spine cross the MediaBox edge and
    # savefig_fixed (no tight crop) clipped it.)
    for ax_, dx in ((axes[1], -0.015), (axes[2], 0.003)):
        bb = ax_.get_position()
        ax_.set_position([bb.x0 + dx, bb.y0, bb.width, bb.height])
    S = FIG43_SCALE

    for label, _, color, ls, marker in ARMS:
        if label not in data:
            continue
        seeds = sorted(data[label])
        for ax, col in ((axes[0], "lambda_max"), (axes[1], "lhg_bs_mean")):
            arrs = np.stack([data[label][s][col].to_numpy() for s in seeds])
            steps = data[label][seeds[0]]["step"].to_numpy()
            # R30: per-seed traces are TEXTURE, not subject — one faint
            # layer per arm (alpha from ST); R39: they inherit the arm
            # LINESYLE (marker-free) as the grayscale-redundant channel.
            for k in range(arrs.shape[0]):
                ax.plot(steps, arrs[k], color=color, ls=ls, lw=ST["seed"],
                        alpha=ST["alpha_seed"], zorder=2)
            # bold seed mean in the semantic method color: fig-32 parity
            # smoothing (240-step log-domain kernel, init spike raw); R39
            # adds the arm marker + markevery (~10 markers per trajectory).
            mean = smooth32(steps, arrs.mean(0))
            ax.plot(steps, mean, color=color, ls=ls, marker=marker,
                    ms=ST["marker"], markevery=max(1, len(steps) // 10),
                    lw=ST["primary"], label=label, zorder=3)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, 80)
    panel_label(axes[0], "a", r"Fixed reference $\lambda_{\mathrm{ref}}$")
    panel_label(axes[1], "b", r"Batch sharpness $\lambda_{\mathrm{step}}$")
    for ax in axes[:2]:
        # late-window marker + theory references, neutral grays
        ax.axvline(args.late_step, color=C["theory_light"], ls=":",
                   lw=ST["theory"], zorder=1)
        ax.axhline(HB, color=C["theory"], ls="--", lw=ST["theory"], zorder=1)
        ax.axhline(GD, color=C["theory"], ls=":", lw=ST["theory"], zorder=1)
        ax.set_xlabel("step")
    # R30.1 label clearance (observer 顺手项, 30_cifar precedent
    # analyze_cifar100_control.py: tag at the line end where the band is
    # empty): the (a) GD tag moves to the LEFT END BELOW its rule — above
    # it the decaying init transient + seed texture sweep the whole 21-30
    # band, below it the rendered layer is verified empty (min mean
    # clearance 12.7 pt, faint-seed 2.6 pt, rule gap 2.4 pt).  The (a) HB
    # tag and the (b) GD tag STAY: an exhaustive anchor scan (left /
    # center / right x every height off both rules, distances measured
    # against the exact mean+seed polylines) found no net-clearance slot —
    # above-HB in (a) is crossed by the transient band at every x, and in
    # (b) below-20 is occupied by SAM while 21-30 is the pinned cluster;
    # the only ink-free band (just under HB=38) would make the "2/η" tag
    # visually ride the HB rule.  Positions below are unchanged for those.
    # (a) HB tag moves to the RIGHT end above its rule (user round: the old
    # left-end tag sat on the init transient — post-smoothing the right end
    # above 38 is verified empty, curves live at 11–25 there). The (a) GD
    # tag keeps its verified below-left slot; (b) gets both rules
    # direct-tagged away from the raw init-spike descent (GD below-rule at
    # x≈1500, clear of the late-step vline at 0.4), plus the family-standard
    # clip flag: the raw init spike reaches λ_step ≈ 250 and is deliberately
    # clipped by the (0, 80) window that protects the informative 10–40 band.
    # (user round: the two threshold rules LEAVE the panels and join the
    # bottom legend — their dashed/dotted handles are self-explanatory
    # there, so neither panel needs direct rule tags anymore. The only
    # remaining annotation is the (b) clip flag: the raw init spike reaches
    # λ_step ≈ 530 and is deliberately clipped by the (0, 80) window that
    # protects the informative 10–40 band.)
    # R38 (user): white stroke halo (same treatment as fig 40/41 boundary
    # labels) instead of the old white bbox pill — curves pass BEHIND the
    # text with no visible rectangle edge.
    bs_max = max(float(data[lab][s]["lhg_bs_mean"].max())
                 for lab in data for s in data[lab])
    axes[1].annotate(rf"init spike $\sim\!{bs_max:.0f}$ (clipped)",
                     (0.96, 0.96), xycoords="axes fraction",
                     ha="right", va="top", ma="right", fontsize=8.0 * S,
                     color="0.25",
                     path_effects=[pe.withStroke(linewidth=3.5,
                                                 foreground="white")])
    # one shared legend, ONE ROW below all three panels (fig-33 pattern):
    # the five arms PLUS the two threshold rules extracted from the panels
    # (user round: 虚线/点线提取出来放到图例中) — the legend row now reads
    # [5 arms | 2/η rule | HB rule] and no direct rule tags remain.
    handles, labels = axes[0].get_legend_handles_labels()
    # rule handles MUST match the axhline styles: GD=20 → dotted, HB=38 →
    # dashed (order paired with the labels below)
    handles += [Line2D([0], [0], color=C["theory"], ls=":", lw=ST["theory"]),
                Line2D([0], [0], color=C["theory"], ls="--", lw=ST["theory"])]
    # R39 terminology (fixplan_r15 §7.1): the HB rule gets the explicit
    # "HB ref." prefix marking it as a fixed reference, not a constraint.
    labels += [rf"$2/\eta = {GD:.0f}$",
               rf"HB ref. $2(1{{+}}\mu)/\eta = {HB:.0f}$"]
    fig.legend(handles, labels, loc="lower center",
                # R39: the mandated "HB ref." prefix lengthens the row past
                # the frozen canvas (measured overflow ~29 px at 100 dpi) —
                # columnspacing 0.7->0.45, handlelength 1.4->1.3 and the
                # anchor 0.53->0.505 re-fit it with L/R margins 14/7 px
                # (the pre-R39 right margin was 4 px). Vertical anchor and
                # the R38 ~6pt key gap are untouched.
                bbox_to_anchor=(0.505, 0.006), ncol=7, frameon=False,
                fontsize=8.0 * S, handlelength=1.3, handletextpad=0.35,
                columnspacing=0.45)

    # (c) late-window paired Δλstep = arm − SGD on shared seeds (the
    # script's existing pairing), forest layout: methods on the y axis,
    # paired Δ on the x axis, zero as a thin theory rule. Per-seed dots
    # open; mean + 95% CI as filled marker with caps (R27: the ordinary
    # paired difference renders at t.ppf(0.975); the TOST layer in the
    # stdout/CSV keeps its 90% CI). NOTE (R30): this plots the BATCH-
    # SHARNESS pairing (traj[...][1]) — matching the panel's own labels
    # and the paper caption's quoted ΔSAM−SGD = −4.30 [−6.70, −1.89];
    # the pre-R30 working-tree version indexed traj[...][2] (λ_ref) and
    # silently contradicted its own title/ylabel.
    axc = axes[2]
    axc.axvline(0.0, color=C["theory"], lw=ST["theory"], zorder=1)
    xlo, xhi = 0.0, 0.0
    ylabels = []
    nrows = sum(1 for l, *_ in ARMS if l != "SGD" and l in data)
    y = nrows - 1
    for label, _, color, _, marker in ARMS:
        if label == "SGD" or label not in data:
            continue
        shared = sorted(set(data[label]) & set(data["SGD"]))
        if len(shared) < 2:
            continue
        d = np.array([traj[label][1][traj[label][0].index(s)]
                      for s in shared]) - \
            np.array([traj["SGD"][1][traj["SGD"][0].index(s)]
                      for s in shared])
        m, h, lo, hi = paired_ci(d, level=0.95)   # R27: 95% CI (t.ppf(0.975))
        jit = np.linspace(-0.16, 0.16, len(shared)) if len(shared) > 1 else [0.0]
        axc.scatter(d, y + jit, s=33, facecolor="none", edgecolors=color,
                    linewidths=1.0 * S, zorder=3)
        axc.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt=marker,
                     ms=ST["marker"] * 1.25, color=color, mew=1.0 * S,
                     elinewidth=ST["errbar"], capsize=ST["capsize"],
                     markeredgecolor=color, zorder=4)
        # n=3 is common to every arm — extracted from the per-row labels
        # (user round) into the shared xlabel; rank-1 labels wrap to two
        # lines so panel (b)/(c) keep their clearance
        xlo = min(xlo, lo, float(d.min()))
        xhi = max(xhi, hi, float(d.max()))
        ylabels.append(label.replace("rank-1 ", "rank-1\n"))
        y -= 1
    pad = 0.09 * (xhi - xlo)
    axc.set_xlim(xlo - pad, xhi + 1.6 * pad)
    axc.set_ylim(-0.6, nrows - 0.4)
    axc.set_yticks(range(nrows))
    axc.set_yticklabels(list(reversed(ylabels)))
    axc.set_xlabel(r"paired $\Delta\lambda_{\rm step}$ (95\% CI, $n{=}3$)")
    panel_label(axc, "c", r"Paired $\Delta$ vs SGD")

    # R39 atomic regeneration (another session may be compiling the paper):
    # save to <name>.tmp.pdf first, mv over the live PDF only after success.
    tmp = savefig_fixed(fig, "43_bsharp_arms.tmp")
    out = tmp.with_name("43_bsharp_arms.pdf")
    tmp.replace(out)
    print(f"=> {out}")


if __name__ == "__main__":
    main()
