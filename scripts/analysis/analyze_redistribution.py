"""Spectral-redistribution diagnostic — review round 2.

The at-ResNet-scale inertness narrative says rank-1 damping leaves λ_max
unchanged because "sharpness redistributes across the rank-(n-1)
complement". This script measures that claim directly from the
spectrum-instrumented runs (rankkx d1 + sgd spec baselines,
--spectrum-k 10 outcome logging):

  (a) top-10 eigenvalue trajectories, seed-mean, SGD vs rank-1 damping:
      does the damped direction drop and another take over λ_max?
  (b) controller-direction attribution: |cos(v̂_ctrl, v_j(outcome))| —
      which outcome eigendirection holds the controller's damped v̂_1
      (ctrl1_top_out_index), and the max overlap.
  (c) eigenspace turnover: min principal-angle cosine between consecutive
      measurement top-k subspaces (spec_turnover_min_cos) — is the top-10
      subspace stable (SGD) or rotating (damping)?

Outputs: printed summary + figures/36_spectral_redistribution.pdf.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steos.analysis.figure_style import C, ST, setup, savefig

LATE_STEP = 2000
KS_SPEC = 10
DAMP_PFX = "cifar100_r18_rankkx_d1_s"
SGD_PFX = "cifar100_r18_sgd_detarpack_spec_s"


def load_family(prefix: str) -> tuple[list[pd.DataFrame], list[int]]:
    dfs, seeds = [], []
    for d in sorted(Path("runs").glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if not s.isdigit() or not (d / "metrics.csv").exists():
            continue
        df = pd.read_csv(d / "metrics.csv")
        cols = [f"lambda_spec{i}" for i in range(1, KS_SPEC + 1)]
        if not all(c in df.columns for c in cols):
            continue
        dfs.append(df)
        seeds.append(int(s))
    return dfs, seeds


def main():
    # R30: source canvas is oversized and shrunk at \includegraphics time;
    # setup(scale=k) puts final on-paper fonts/strokes at the R30 baseline
    # (k = 1/s from figure_style.report(), replacing the round-29 manual
    # rcParams back-computation below, which has been retired).
    setup(scale=1.2648)
    damp, ds = load_family(DAMP_PFX)
    sgd, ss = load_family(SGD_PFX)
    print(f"rank-1 damping (spectrum): seeds {ds}")
    print(f"SGD (spectrum):             seeds {ss}")
    if not damp or not sgd:
        raise SystemExit("need at least one spectrum run per family "
                         "(queue stage 3)")

    def spec_matrix(dfs):
        # (n_seeds, n_steps, KS_SPEC) aligned on the common step grid
        steps = dfs[0]["step"].to_numpy()
        arr = np.stack([df[[f"lambda_spec{i}" for i in
                            range(1, KS_SPEC + 1)]].to_numpy() for df in dfs])
        return steps, arr

    steps_d, A_d = spec_matrix(damp)
    steps_s, A_s = spec_matrix(sgd)

    # ---- summary numbers ---------------------------------------------------
    late_d = steps_d >= LATE_STEP
    late_s = steps_s >= LATE_STEP
    for name, arr, late in (("SGD", A_s, late_s), ("rank-1 damp", A_d, late_d)):
        l1 = arr[:, late, 0].mean()
        l2 = arr[:, late, 1].mean()
        print(f"{name:12s} late λ1 = {l1:6.2f}  λ2 = {l2:6.2f}  "
              f"λ2/λ1 = {l2/l1:.3f}  (mean over seeds x steps)")
    if all("ctrl1_top_out_index" in df.columns for df in damp):
        idx = np.concatenate([df.loc[df["step"] >= LATE_STEP,
                                     "ctrl1_top_out_index"].to_numpy() for df in damp])
        cos1 = np.concatenate([df.loc[df["step"] >= LATE_STEP,
                                      "cos_ctrl1_v1"].abs().to_numpy() for df in damp])
        cmax = np.concatenate([df.loc[df["step"] >= LATE_STEP,
                                      "ctrl1_top_out_cos"].to_numpy() for df in damp])
        top5 = np.concatenate([df.loc[df["step"] >= LATE_STEP,
                                      "ctrl1_top_out_index"].le(5).to_numpy() for df in damp])
        print(f"controller v̂1 == outcome v1: {float((idx == 1).mean()):.2f} of late "
              f"measurements; within top-5: {float(top5.mean()):.2f}")
        print(f"|cos(v̂_ctrl1, v1_out)| median {np.median(cos1):.3f}  "
              f"max-over-j median {np.median(cmax):.3f}")
    for name, dfs in (("SGD", sgd), ("rank-1 damp", damp)):
        if "spec_turnover_min_cos" in dfs[0].columns:
            t = np.concatenate([df.loc[df["step"] >= LATE_STEP,
                                       "spec_turnover_min_cos"].dropna().to_numpy()
                                for df in dfs if df["spec_turnover_min_cos"].notna().any()])
            print(f"{name:12s} top-10 subspace turnover (min cos, consecutive "
                  f"measurements): median {np.median(t):.3f}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3),
                             gridspec_kw=dict(wspace=0.42))
    ax = axes[0]
    # R30: the per-index spectrum gradient language is UNCHANGED (continuous
    # scalar encoding keeps its sequential colormap); only stroke hierarchy
    # is applied (λ1 primary, λ2+ secondary). Threshold ref at theory weight.
    # R39 grayscale-redundancy lane (fixplan_r15 §6): the six per-index
    # curves carry DISTINCT linestyles so the spectrum ordering stays
    # readable in grayscale print. No markers (too dense at this sampling).
    # NOTE: the fixplan's width ramp [2.0, 1.8, 1.2 x4] collapses onto the
    # locked R38 ST tiers — primary (1.5 pt final) for λ1, secondary
    # (1.2 pt final) for λ2–λ6; the λ1/λ2-vs-tail grouping rides on the
    # linestyle channel instead.
    cmap = plt.get_cmap("viridis")
    SPEC_LS = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
    for j in range(min(6, KS_SPEC)):
        ax.plot(steps_s, A_s[:, :, j].mean(axis=0), color=cmap(j / 6),
                ls=SPEC_LS[j],
                lw=ST["primary"] if j == 0 else ST["secondary"],
                label=fr"$\lambda_{{{j+1}}}$")
    ax.axhline(38, color=C["theory_light"], ls=":", lw=ST["theory"])
    # R47c: log λ axis — the six-curve fan (λ1 ~600 transient → λ6 ~10)
    # spans ~1.8 decades, so no explicit ylim is needed for readability.
    ax.set_yscale("log")
    ax.set_title(r"\textbf{(a)} SGD: top-6 spectrum", color="#1A1D21")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\lambda_i$"); ax.legend(fontsize=10.12, ncol=2)
    ax = axes[1]
    for j in range(min(6, KS_SPEC)):
        ax.plot(steps_d, A_d[:, :, j].mean(axis=0), color=cmap(j / 6),
                ls=SPEC_LS[j],
                lw=ST["primary"] if j == 0 else ST["secondary"],
                label=fr"$\lambda_{{{j+1}}}$")
    ax.axhline(38, color=C["theory_light"], ls=":", lw=ST["theory"])
    # R47c: log λ axis, matching panel (a).
    ax.set_yscale("log")
    ax.set_title(r"\textbf{(b)} Rank-1 damping: top-6", color="#1A1D21")
    ax.set_xlabel("step"); ax.legend(fontsize=10.12, ncol=2)
    ax = axes[2]
    if all("ctrl1_top_out_index" in df.columns for df in damp):
        hist_x = np.arange(1, KS_SPEC + 1)
        counts = np.bincount(
            np.concatenate([df.loc[df["step"] >= LATE_STEP,
                                   "ctrl1_top_out_index"].astype(int).to_numpy()
                            for df in damp]), minlength=KS_SPEC + 1)[1:]
        # R30: the single-hue default-blue bar is replaced by the damping
        # family hue with within-family hierarchy — j=1 (the damped
        # direction re-emerging as top) at full strength, the tail at 0.45
        # alpha; bar form itself is unchanged this round.
        from matplotlib.colors import to_rgba
        bar_colors = [to_rgba(C["damping"], 1.0 if j == 1 else 0.45)
                      for j in hist_x]
        ax.bar(hist_x, counts / counts.sum(), color=bar_colors)
        ax.set_xlabel(r"outcome eigenindex $j$")
        ax.set_ylabel("fraction of late measurements")
        ax.set_title(r"\textbf{(c)} Damped-direction rank", color="#1A1D21")
    # explicit layout: wspace keeps the enlarged three titles clear of each
    # other (tight_layout ignores title-width overflow across panels)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.2, top=0.83,
                        wspace=0.42)
    # R39 atomic regeneration (another session may be compiling the paper):
    # save to <name>.tmp.pdf first, mv over the live PDF only after success.
    tmp = savefig(fig, "36_spectral_redistribution.tmp")
    out = tmp.with_name("36_spectral_redistribution.pdf")
    tmp.replace(out)
    print(f"=> {out}")


if __name__ == "__main__":
    main()
