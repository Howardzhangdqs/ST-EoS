r"""Fig (fig:bridge) — clamp-free bridge on CIFAR-100/ResNet-18 (n=5, deterministic).

Finding 28 (FINDINGS_zh.md): the C3-at-scale inertness is a real phenomenon,
not a clamp artifact. This figure backs the "待补" placeholder referenced from
paper/main.tex via IfFileExists. Two panels:

  (a) lambda_hat_grad (per-model deterministic gradient-Rayleigh proxy
      $\hat{\lambda}_g$) trajectories for sgd / rank1 (clamped C3) /
      rank1_noclamp (clamp-free C3, fires every step) / sam: bold cross-seed
      mean lines (lw 2) smoothed with a Gaussian filter in the log domain
      (σ=12 samples ≈ 240 steps, reflect edges); the two-sample init
      transient (steps 0–40) is kept unsmoothed, with a log-linear
      crossfade over samples 2–12 (steps 40–240) so the bold line still
      enters the panel from the clipped top edge. Each arm is underlaid
      with the UNsmoothed cross-seed mean
      as a faint thin line (alpha 0.16, lw 0.45). Smoothing is a
      display-layer only — all statistics (late-window table, panel (b))
      use the raw per-seed means. The former per-step ±95% CI band was
      removed from this panel; uncertainty now lives only in panel (b)'s
      error bars. Arms are distinguished by linestyle + hue: the four hues
      carry distinct grayscales (luminance 0.39/0.44/0.44/0.58) and the
      two damping-blue rank-1 arms (same gray) separate by linestyle
      (clamped dashed / clamp-free dash-dot), so arms separate in grayscale
      and for red/green colorblind viewers; staggered zorders keep any arm
      from fully overplotting another. Log y-axis so the FULL range is in-frame —
      the init transient ($\hat{\lambda}_g \approx 500$ at step 0) and the
      late regime (~10-25) both visible, no off-scale clipping (same log-axis
      convention as the trajectory panels of Figs. 28/30).

  (b) Late-window (step >= 2000) $\hat{\lambda}_g$ mean ± 95% CI bars,
      n=5 seeds (seeds 0-4; s3/s4 added by the bridge seed-scaling rerun).
      Reference values = the n=5 rerun itself: sgd 18.73±1.29, rank1
      18.13±1.47, rank1_noclamp 17.14±1.21, sam 12.91±0.59 (supersedes the
      n=3 Finding-28 table: 18.18±1.15 / 18.12±0.97 / 17.41±1.30 / 12.78±0.67;
      ordering and conclusions unchanged). CI convention at n=5: Student-t
      df=n-1 (t=2.776) on per-seed late-window means — the exact-interval
      convention used by analyze_multiseed_control.py's ci(); the old n=3
      table used the 1.96·sd/sqrt(n) normal approximation (t would have been
      1.42x wider there; at n=5 the two conventions sit much closer).

Reads:  runs/cifar100_r18_{sgd,rank1,rank1_noclamp,sam}_s{0..4}/metrics.csv
Writes: figures/32_bridge_clampfree_c3.pdf
"""
from __future__ import annotations

import math
from pathlib import Path

from steos.analysis.figure_style import (
    C,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    ST,
    setup,
    savefig,
)

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

RUNS = Path("runs")

SEEDS = [0, 1, 2, 3, 4]
LATE_STEP = 2000          # late-window definition (matches analyze_cifar100_control.py)
COL = "lambda_hat_grad"   # λ̂_g: gradient Rayleigh-quotile proxy (deterministic, per-model)
# Student-t 95% multiplier by sample size n (df = n-1) — same convention as
# analyze_multiseed_control.py's ci(). At n=5: 2.776 (the old n=3 table used
# the 1.96 normal approximation).
T95 = {3: 4.303, 4: 3.182, 5: 2.776}

# opt -> (display name, color, linestyle, zorder). Colors come from
# the global METHOD_COLORS registry (same method = same color in every paper
# figure — hues unchanged); linestyle comes from the companion registry,
# a redundant non-color channel so the two damping-blue rank-1 arms separate
# in grayscale (red/green adjacency is colorblind-unfriendly — reviewer
# request). zorders staggered so no arm fully overplots another.
ARMS = [
    ("sgd",           "SGD",                 METHOD_COLORS["sgd"],
     METHOD_LINESTYLES["sgd"],           3),
    ("rank1",         "rank-1 (clamped)",    METHOD_COLORS["rank1"],
     METHOD_LINESTYLES["rank1"],         4),
    ("rank1_noclamp", "rank-1 (clamp-free)", METHOD_COLORS["rank1_noclamp"],
     METHOD_LINESTYLES["rank1_noclamp"], 5),
    ("sam",           "SAM",                 METHOD_COLORS["sam"],
     METHOD_LINESTYLES["sam"],           6),
]

# Late-window reference numbers (n=5 rerun, seeds 0-4; Student-t CIs) for
# validation. Supersedes the n=3 Finding-28 table, which the s3/s4 seeds
# moved the means off by up to 3% (sgd 18.18 -> 18.73).
PAPER_LATE = {
    "sgd":           (18.73, 1.29),
    "rank1":         (18.13, 1.47),
    "rank1_noclamp": (17.14, 1.21),
    "sam":           (12.91, 0.59),
}


def ci95(x):
    """95% CI half-width, Student-t df=n-1 on per-seed means (t·sd/√n);
    falls back to 1.96 for n outside T95."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return 0.0
    return float(T95.get(x.size, 1.96) * x.std(ddof=1) / math.sqrt(x.size))


def load(opt, seed):
    csv = RUNS / f"cifar100_r18_{opt}_s{seed}/metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv}")
    return pd.read_csv(csv).sort_values("step").reset_index(drop=True)


def main():
    # R30: source canvas is oversized and shrunk at \includegraphics time;
    # setup(scale=k) keeps final on-paper stroke weights/fonts at the R30
    # baseline (k = 1/s from figure_style.report(), measured on the emitted
    # PDF — see the round-30 QA log).
    setup(scale=1.2558)

    data = {opt: [load(opt, s) for s in SEEDS] for opt, *_ in ARMS}

    # ---- late-window stats (per-seed mean -> cross-seed mean ± CI) ----
    late = {}
    print("late-window (step >= %d) %s: computed vs published" % (LATE_STEP, COL))
    ok = True
    for opt, name, *_ in ARMS:
        per_seed = []
        for df in data[opt]:
            v = df.loc[df.step >= LATE_STEP, COL].dropna().to_numpy(float)
            per_seed.append(float(v.mean()) if v.size else float("nan"))
        m = float(np.mean(per_seed))
        c = ci95(per_seed)
        late[opt] = (m, c, per_seed)
        pm, pc = PAPER_LATE[opt]
        dev_m = abs(m - pm) / pm * 100.0
        dev_c = abs(c - pc) / pc * 100.0 if pc > 0 else 0.0
        flag = "" if max(dev_m, dev_c) <= 2.0 else "  <-- MISMATCH >2%"
        ok = ok and (flag == "")
        print(f"  {opt:15s} mine {m:6.2f} ± {c:4.2f}   paper {pm:6.2f} ± {pc:4.2f}"
              f"   dev mean {dev_m:.2f}% / ci {dev_c:.2f}%{flag}")
    if not ok:
        raise SystemExit("late-window numbers deviate >2% from the published table — aborting")

    n = len(SEEDS)
    fig, (ax_traj, ax_bar) = plt.subplots(
        1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )

    # ----- (a) trajectories: cross-seed mean (smoothed bold + raw underlay) -----
    # R30 fix ③ (observer QA): the two rank-1 arms share the damping-blue
    # hue, so the linestyle redundancy is REAL — clamped=dashed,
    # clamp-free=dashdot (paper caption: "arms distinguished by linestyle
    # and hue"). SGD/SAM keep solid strokes (unique hues). Markers were
    # removed (v5): arm identity is carried by hue + linestyle alone.
    LINESTYLE = {"sgd": "-", "rank1": "--", "rank1_noclamp": "-.", "sam": "-"}
    legend_handles = []
    for opt, name, color, _ls, z in ARMS:
        cols = [pd.Series(df[COL].to_numpy(), index=df["step"].to_numpy(), name=f"s{i}")
                for i, df in enumerate(data[opt])]
        wide = pd.concat(cols, axis=1)
        step = wide.index.to_numpy()
        mean = wide.mean(axis=1).to_numpy()
        # faint thin UNDERLAY: the raw (unsmoothed) cross-seed mean at
        # zorder 2, so the smoothed bold line sits on top of it
        ax_traj.plot(step, mean, linestyle="-", lw=0.45, color=color,
                     alpha=0.16, zorder=2)
        # bold smoothed mean: Gaussian in the log domain (sigma=12 samples
        # = 240 steps, reflect edges), with the two-sample init spike
        # (step 0/20, off-frame) kept raw and a log-linear crossfade over
        # samples 2..12 (steps 40..240) so the bold line still enters the
        # panel from the clipped top edge. Display-layer only; every
        # statistic above (late-window table, panel (b) bars) still uses
        # the raw per-seed means.
        lg = np.log10(mean)
        sm_lg = gaussian_filter1d(lg, sigma=12, mode="reflect")
        w = np.clip((np.arange(len(lg)) - 2) / (12 - 2), 0, 1)
        sm = 10 ** ((1 - w) * lg + w * sm_lg)
        ax_traj.plot(step, sm, LINESTYLE[opt], lw=ST["primary"], color=color,
                     zorder=z)
        legend_handles.append(
            Line2D([0], [0], color=color, lw=ST["primary"], ls=LINESTYLE[opt],
                   label=name))
    ax_traj.set_xlim(0, 5000)
    # log y-axis CLIPPED (user-set range 10² down to 6): the full range
    # (init transient band tops at ~727) compressed the informative late
    # region (λ̂g ∈ [10,19]); this window keeps the transient decay in
    # frame while giving the late region most of the panel height — only
    # the spike itself is clipped, flagged by the annotation.
    ax_traj.set_yscale("log")
    ax_traj.set_ylim(6, 100)
    # R30 fix ① (observer QA): the annotation is stacked TWO-LINE so its
    # right end no longer touches the upper-right legend's SGD handle
    # (halved footprint; same words, same anchor).
    ax_traj.annotate(r"init transient" "\n" r"$\sim\!7\times10^{2}$ (clipped)",
                     xy=(0.04, 0.97), xycoords="axes fraction",
                     ha="left", va="top", ma="left", color="0.25",
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    ax_traj.set_xlabel("step")
    ax_traj.set_ylabel(r"$\hat{\lambda}_g$")
    # R30: short panel title (protocol — n seeds, band convention — lives in
    # the paper caption); left-aligned above the axes, tag preserved.
    ax_traj.set_title(rf"\textbf{{(a)}} Trajectories ($n{{=}}{n}$)", color="#1A1D21")
    ax_traj.legend(handles=legend_handles, loc="upper right")
    ax_traj.set_axisbelow(True)   # grid behind data ink (trajectory panel)
    # R30: trajectory panels carry NO grid (only bar/point summaries may add
    # a faint horizontal one — see panel (b)).

    # ----- (b) late-window bars: mean ± 95% CI -----
    x = np.arange(len(ARMS))
    means = [late[opt][0] for opt, *_ in ARMS]
    cis = [late[opt][1] for opt, *_ in ARMS]
    colors = [c for _, _, c, _, _ in ARMS]
    # OPAQUE bars (no alpha) + axisbelow: grid renders BEHIND the bars
    # (default grid zorder 1.5 > bar zorder 1 drew lines on top / showed
    # through the semi-transparent patches). Colors/edges unchanged.
    ax_bar.bar(x, means, yerr=cis, capsize=ST["capsize"], width=0.62,
               color=colors, edgecolor="black", lw=0.7)
    for xi, (m, c) in enumerate(zip(means, cis)):
        # two-line form (mean / ±CI): adjacent bars have near-equal values
        # (18.18/18.12/17.41) and the single-line labels collided horizontally
        ax_bar.annotate(fr"${m:.2f}$" "\n" fr"$\pm {c:.2f}$",
                        xy=(xi, m + c), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontweight="bold")
    ax_bar.set_xticks(x)
    # two-line horizontal category labels — no numeric tick labels, no xlabel;
    # stacking the long clamp-free name vertically avoids the collisions the
    # rotated single-line form produced.
    ax_bar.set_xticklabels(["SGD", "rank-1\nclamped", "rank-1\nclamp-free", "SAM"],
                           rotation=0, ha="center")
    ax_bar.set_ylabel(r"late-window $\hat{\lambda}_g$")
    ax_bar.set_ylim(0, 26)
    ax_bar.set_title(r"\textbf{(b)} Late-window mean", color="#1A1D21")
    ax_bar.set_axisbelow(True)   # grid zorder 0.5, behind the opaque bars
    # R30: bar summary -> faint horizontal major grid only (neutral gray,
    # alpha <= 0.25), no vertical grid.
    ax_bar.grid(axis="y", alpha=0.25, color=C["theory_light"])

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)  # headroom for the two-line category labels
    out = savefig(fig, "32_bridge_clampfree_c3")
    print(f"\n[fig] {out}")


if __name__ == "__main__":
    main()
