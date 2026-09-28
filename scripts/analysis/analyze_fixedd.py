"""Aggregate + figure for the fixed-divisor causal control (review round 4).

Reads   figures/fixedd_{summary,traj}_s{0..4}.csv  (fixedd_experiment.py)
Writes  figures/42_fixedd.pdf + figures/fixedd_summary_agg.csv

Two headline quantities:
  (1) EQUILIBRIUM TRACKING — late-window (step >= 1500) mean raw λ_max per
      (arm, seed), aggregated mean±95%CI over the non-diverged seeds, against
      the 2-D point prediction λ_eq = 2·d/η (valid because γ = 1−1/d is
      CONSTANT: the fixed-point condition is not the adaptive rule's identity
      y* = y).
  (2) FEEDBACK-FREE MATCH — paired per-seed difference between the fixed
      d=6.5 arm and the adaptive (self-gating) damping arm on the seeds where
      the fixed arm survives: same equilibrium without any feedback, but
      conditionally stable (n_survive/5 vs the adaptive arm's 5/5).

Panel (a) seed-mean λ_max trajectories (log y) with the 2d/η guide lines;
panel (b) equilibrium vs d: mean±CI points, the prediction line 2d/η, the
adaptive arm plotted at its realized mean divisor, divergence fractions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import setup  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
LATE_STEP = 1500
LR = 0.05
TWO_OVER_ETA = 2.0 / LR          # 40
ARMS = [1.25, 1.5, 2.0, 2.5, 3.0, 6.5]

# Round-4 reviewer ask: a redundant NON-COLOR coding channel (grayscale /
# colorblind-safe), mirroring the ARMS treatment in analyze_bsharp_arms.py
# and the registry conventions in steos/analysis/figure_style.py.  cmap
# hues below stay the primary channel; each fixed-d arm additionally gets
# a UNIQUE marker + linestyle pair so the ~8-entry legend no longer leans
# on near-identical dash patterns.  References take offset markers (P / *).
ARM_STYLE = {                      # d -> (marker, linestyle)
    1.25: ("o", "-"),
    1.5:  ("s", "--"),
    2.0:  ("^", "-."),
    2.5:  ("D", (0, (5, 2))),
    3.0:  ("v", (0, (3, 1, 1, 1))),
    6.5:  ("x", ":"),
}
REF_STYLE = {"sgd": ("P", "--"), "rankk_k1": ("*", (0, (5, 2)))}


def arm_name(d: float) -> str:
    return f"fixedd_d{d:g}"


def ci95(xs):
    xs = np.asarray(xs, float)
    n = len(xs)
    if n < 2:
        return float(xs.mean()), 0.0
    h = float(stats.t.ppf(0.975, n - 1) * xs.std(ddof=1) / np.sqrt(n))
    return float(xs.mean()), h


def load():
    trajs, summaries = [], []
    for s in SEEDS:
        p_t, p_s = Path(f"figures/fixedd_traj_s{s}.csv"), Path(f"figures/fixedd_summary_s{s}.csv")
        if not (p_t.exists() and p_s.exists()):
            raise SystemExit(f"missing {p_t} / {p_s} — run fixedd_experiment.py --seed {s} first")
        trajs.append(pd.read_csv(p_t).assign(seed=s))
        summaries.append(pd.read_csv(p_s).assign(seed=s))
    return pd.concat(trajs, ignore_index=True), pd.concat(summaries, ignore_index=True)


def late_stats(traj: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, seed): late-window mean/min/max of the ground-truth λ_max,
    NaN fraction (divergence), and the sensed λ̂₁ mean."""
    rows = []
    for (opt, seed), g in traj.groupby(["opt", "seed"]):
        late = g[g["step"] >= LATE_STEP].sort_values("step")
        lam = late["lambda_max"].to_numpy(float)
        n_nan = int(np.isnan(lam).sum())
        lam_ok = lam[~np.isnan(lam)]
        rows.append(dict(
            opt=opt, seed=seed,
            late_mean=float(lam_ok.mean()) if lam_ok.size else np.nan,
            late_min=float(lam_ok.min()) if lam_ok.size else np.nan,
            late_max=float(lam_ok.max()) if lam_ok.size else np.nan,
            n_measure=len(lam), n_nan=n_nan,
            diverged=bool(n_nan > 0),
            lam_hat1_mean=float(np.nanmean(late["lam_hat1"].to_numpy(float))),
        ))
    return pd.DataFrame(rows)


def main():
    setup()
    # Round-4 readability pass (same treatment as analyze_bsharp_arms.py):
    # the paper renders this figure at 0.70\linewidth (~3.9in of the 5.5in
    # ICLR text width), so the project defaults (10pt titles / 9pt ticks)
    # land ~3.7pt effective — flagged too small by the reviewer.  Bump text
    # sizes ~25-30% and shrink the canvas below accordingly; two-line panel
    # titles + staggered divergence annotations avoid collisions at the new
    # sizes.
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })
    traj, summ = load()
    st = late_stats(traj)

    print(f"=== Fixed-divisor causal control: late window step >= {LATE_STEP}, "
          f"2/η = {TWO_OVER_ETA:g} ===\n")
    rows = []
    for d in ARMS:
        g = st[st.opt == arm_name(d)]
        stable = g[~g.diverged]
        mean, h = ci95(stable["late_mean"].to_numpy()) if len(stable) else (np.nan, np.nan)
        pred = TWO_OVER_ETA * d
        rows.append(dict(
            arm=arm_name(d), d=d, n_stable=len(stable), n_seeds=len(g),
            div_frac=len(g) - len(stable) if len(g) else np.nan,
            late_lmax_mean=mean, ci95=h, predicted=pred,
            ratio=mean / pred if pred and np.isfinite(mean) else np.nan,
            lam_hat1_mean=float(stable["lam_hat1_mean"].mean()) if len(stable) else np.nan,
        ))
    # references (never diverged on this protocol)
    for ref, label in (("rankk_k1", "adaptive d=λ̂/(2/η)"), ("sgd", "SGD")):
        g = st[st.opt == ref]
        mean, h = ci95(g["late_mean"].to_numpy())
        rows.append(dict(arm=ref, d=np.nan, n_stable=len(g), n_seeds=len(g),
                         div_frac=0, late_lmax_mean=mean, ci95=h,
                         predicted=np.nan, ratio=np.nan,
                         lam_hat1_mean=float(g["lam_hat1_mean"].mean())))
    agg = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    # ---- (2) feedback-free match: fixed 6.5 vs adaptive, paired ------------
    ad = st[st.opt == "rankk_k1"].set_index("seed")
    f65 = st[st.opt == arm_name(6.5)].set_index("seed")
    common = [s for s in SEEDS
              if s in f65.index and not f65.loc[s, "diverged"] and s in ad.index]
    if common:
        diffs = [f65.loc[s, "late_mean"] - ad.loc[s, "late_mean"] for s in common]
        dm, dh = ci95(diffs)
        ad_div = float(np.mean([f65.loc[s, "late_mean"] / ad.loc[s, "late_mean"]
                                for s in common]))
        print(f"\n=== feedback-free match: fixed d=6.5 vs adaptive damping ===")
        print(f"  seeds where fixed 6.5 survives: {len(common)}/5 "
              f"(adaptive: 5/5 stable)")
        for s in common:
            print(f"    seed {s}: fixed {f65.loc[s,'late_mean']:8.2f}  "
                  f"adaptive {ad.loc[s,'late_mean']:8.2f}  "
                  f"Δ {f65.loc[s,'late_mean']-ad.loc[s,'late_mean']:+7.2f}")
        print(f"  paired Δ = {dm:+.3f} ± {dh:.3f}  |  "
              f"ratio fixed/adaptive = {ad_div:.4f}")
        # the adaptive arm's realized mean divisor on those seeds
        divmean = float(np.mean([
            traj[(traj.opt == "rankk_k1") & (traj.seed == s)
                 & (traj.step >= LATE_STEP)]["lam_hat1"].mean() / TWO_OVER_ETA
            for s in common]))
        print(f"  adaptive realized mean divisor on those seeds: d̄ = {divmean:.3f}")

    # ---- figure -------------------------------------------------------------
    # 5.0 x 2.4 in (was 7.8 x 3.5): at the paper's fixed 0.50\linewidth render
    # (2.75 in of the 5.5 in ICLR text width) the smaller canvas scales the
    # 12pt-base fonts to ~6.6pt effective (was ~4.3pt -> "(n/5)" counts were
    # illegible).  At this panel size an in-axes legend is panel-sized (8 and
    # 7 entries at 10.5pt), so both legends move to a two-row strip BELOW the
    # panels — clear of every trajectory — with opaque frames.
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4))

    ax = axes[0]
    cmap = {1.25: "#1f77b4", 1.5: "#2ca02c", 2.0: "#d62728", 2.5: "#9467bd",
            3.0: "#8c564b", 6.5: "#e377c2"}
    for d in ARMS:
        g = traj[traj.opt == arm_name(d)]
        g_st = st[st.opt == arm_name(d)]
        n_st = int((~g_st.diverged).sum())          # survived seeds
        n_tot = len(g_st)
        piv = g.pivot_table(index="step", columns="seed", values="lambda_max")
        m = piv.mean(axis=1, skipna=True)
        # stability rides on the fade + the (n_surv/n_tot) label; the
        # linestyle/marker pair now encodes ARM identity (was: dashed for
        # any diverged/partial arm — the near-identical-dash complaint)
        unstable = n_st < n_tot
        mk, ls = ARM_STYLE[d]
        ax.plot(piv.index, m, lw=1.4, color=cmap[d], ls=ls,
                marker=mk, ms=3.5,
                markevery=max(1, len(piv.index) // 8),
                alpha=0.55 if unstable else 1.0,
                label=f"$d{{=}}{d:g}$ ({n_st}/{n_tot})")
        ax.axhline(TWO_OVER_ETA * d, color=cmap[d], ls=":", lw=0.8, alpha=0.7)
    for ref, col, lab in (("sgd", "k", "GD"), ("rankk_k1", "tab:gray", "adaptive $d{=}\\hat\\lambda/(2/\\eta)$")):
        g = traj[traj.opt == ref]
        piv = g.pivot_table(index="step", columns="seed", values="lambda_max")
        mk, ls = REF_STYLE[ref]
        ax.plot(piv.index, piv.mean(axis=1), lw=1.6, color=col, ls=ls,
                marker=mk, ms=4.5, markevery=max(1, len(piv.index) // 8),
                label=lab)
    ax.axhline(TWO_OVER_ETA, color="k", ls="-", lw=0.8, alpha=0.4)
    ax.set_yscale("log")
    # headroom: the diverging d=2.5 mean spikes to ~1.6e3 at step 300 and the
    # auto top put its apex ~5% below the top spine (read as clipping); lift
    # the ceiling so the spike detaches cleanly.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.6)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_{\mathrm{raw}}$ (log)")
    ax.set_title("(a) fixed-$d$ damping:\nraw $\\lambda_{\\mathrm{raw}}$ vs $2d/\\eta$ (dotted)")
    handles_a, labels_a = ax.get_legend_handles_labels()

    ax = axes[1]
    ds = np.linspace(1.0, 7.2, 50)
    ax.plot(ds, TWO_OVER_ETA * ds, color="k", lw=1.2,
            label=r"prediction $\lambda_{\rm eq}=2d/\eta$")
    sub = agg[agg.arm.str.startswith("fixedd")]
    ann_ys = (312, 218)           # diverged-arm labels stacked in the empty
    n_ann = 0                     # upper-left band (x fixed, clear of the
    ann_x = 1.55                  # prediction line + errorbar cluster; the
    for _, r in sub.iterrows():   # colored text says which d it labels)
        if np.isfinite(r.late_lmax_mean):
            ax.errorbar(r.d, r.late_lmax_mean, yerr=r.ci95,
                        marker=ARM_STYLE[r.d][0], ms=5, ls="none",
                        capsize=3, color=cmap[r.d],
                        label=f"$d{{=}}{r.d:g}$ ({r.n_stable}/5)")
        else:
            ax.annotate(f"$d{{=}}{r.d:g}$:\n{int(r.div_frac)}/5 div.",
                        (ann_x, ann_ys[n_ann % len(ann_ys)]), fontsize=9,
                        color=cmap[r.d], ha="center", va="top")
            n_ann += 1
    adp = agg[agg.arm == "rankk_k1"].iloc[0]
    ax.errorbar(6.72, adp.late_lmax_mean, yerr=adp.ci95, marker="*", ms=7,
                mfc="none", color="tab:gray", capsize=3, ls="none",
                label=r"adaptive ($\bar{d}\approx 6.7$), 5/5")
    sg = agg[agg.arm == "sgd"].iloc[0]
    ax.errorbar(1.0, sg.late_lmax_mean, yerr=sg.ci95, marker="P", ms=6.5,
                mfc="none", color="k", capsize=3, ls="none",
                label=f"GD (pin), 5/5")
    ax.set_xlabel(r"fixed divisor $d$")
    ax.set_ylabel(r"late $\lambda_{\mathrm{raw}}$ (step $\geq$ 1500)")
    ax.set_title("(b) equilibrium tracks $2d/\\eta$\nwhere the orbit holds")
    ax.set_xlim(0.8, 7.5)
    ax.set_ylim(0, 325)           # headroom for the floating div. labels
    handles_b, labels_b = ax.get_legend_handles_labels()

    fig.tight_layout()
    # Two-row legend strip below the panels (the in-axes legends were
    # panel-sized at this canvas and sat on top of trajectories).  Opaque
    # frames, no fancybox; entries keep the exact labels/markers from above.
    leg_kw = dict(fontsize=10.5, frameon=True, framealpha=1.0, fancybox=False,
                  edgecolor="0.6", handlelength=1.3, handletextpad=0.4,
                  labelspacing=0.4, borderpad=0.4, columnspacing=0.9)
    fig.legend(handles_a, labels_a, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.055), **leg_kw)
    fig.legend(handles_b, labels_b, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.185), **leg_kw)
    out = Path("figures/42_fixedd.pdf")
    fig.savefig(out, bbox_inches="tight")
    agg.to_csv("figures/fixedd_summary_agg.csv", index=False)
    print(f"\n=> {out} + figures/fixedd_summary_agg.csv")


if __name__ == "__main__":
    main()
