"""Aggregate + figures for the η×d collapse and the delayed-onset fixed-d
control (review round 5).

Reads (new, this round):
    figures/etad_collapse_s{0..4}.csv     scripts/experiments/etad_collapse_experiment.py
    figures/delayed_fixedd_s{0..4}.csv    scripts/experiments/delayed_fixedd_experiment.py
Reads (prior round, READ-ONLY — never overwritten):
    figures/fixedd_traj_s{0..4}.csv, figures/fixedd_summary_s{0..4}.csv
    (for the η=0.05, d=6.5 phase-diagram / collapse point)

Writes (R30 data-layer guard: only when MISSING — published summary CSVs
are read back and never rewritten, per the zero-change red line):
    figures/etad_collapse_summary.csv
    figures/delayed_fixedd_summary.csv
    figures/47_etad_collapse.pdf     UNIFIED 3-panel main-text figure, R30
                                     restyle on the FROZEN footprint
                                     766.872 x 300.092 pt (savefig_fixed,
                                     no tight crop): (a) η·λ̄_late/2 vs d
                                     (on-orbit cells filled, the two
                                     off-identity η=0.025 survivors open,
                                     single damping hue + per-η marker
                                     shapes; muted 5/5 counts, only
                                     sub-full survival emphasized);
                                     (b) survival heatmap n_surv/5,
                                     cividis kept, narrow colorbar;
                                     (c) representative λ_raw(t) with
                                     line-end direct labels (param +
                                     short gray regime word), predicted
                                     levels 2d/η as thin theory-gray
                                     references drawn over the data span
                                     only
    figures/48_fixedd_phase.pdf      phase diagram (appendix), R30
                                     apply-only restyle (strokes / colors
                                     / spine / shortened title; layout,
                                     canvas and tight save unchanged):
                                     x=d, one row per η, survival fraction
                                     as cividis color, annotated η·λ̄/2

Late-window convention: step >= 1500 mean of the ground-truth λmax
(measurements every 100 steps), identical to analyze_fixedd.py.
Survivor = not diverged AND no NaN in the late λmax window.
On-orbit = surviving cell with ratio (late mean / (2d/η)) >= 0.9.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from steos.analysis.figure_style import (
    C, ST, FROZEN_PAGES, INCLUDE_FRAC, TEXTWIDTH_PT,
    setup, savefig, savefig_fixed, frozen_figsize, panel_label,
)

SEEDS = (0, 1, 2, 3, 4)
LATE_STEP = 1500
ETAS = [0.025, 0.05, 0.1]
DS = [1.25, 1.5, 2.0, 2.5, 3.0]
PRIOR_D65 = 6.5          # η=0.05 anchor from the prior fixed-d round
LR_PRIOR = 0.05
# prior-round anchors for validation (a): figures/fixedd_summary_agg.csv
PRIOR_ANCHORS = {1.25: 47.8461954395255, 1.5: 59.23760931979857,
                 2.0: 74.72082342952142}

# R30 visual system. Every plotted series here belongs to ONE method family
# (the fixed-d rank-1 damping protocol), so it gets ONE hue: damping blue.
# The η dimension is coded by marker SHAPE (never color-only); on- vs
# off-identity by filled vs open markers (the redundant non-color channel).
# References (identity, predicted 2d/η, escape ceiling) are neutral theory
# grays. Continuous scalar encodings (survival fraction) keep cividis.
FIG47_SCALE = (FROZEN_PAGES["47_etad_collapse"][0]
               / (INCLUDE_FRAC["47_etad_collapse"] * TEXTWIDTH_PT))  # ≈2.3335


def _mix_white(hex_color: str, frac: float) -> str:
    """Blend a #RRGGBB color toward white by frac (fig-33 shade ramp)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    up = lambda c: round(c + (255 - c) * frac)
    return "#{:02x}{:02x}{:02x}".format(up(r), up(g), up(b))
ETA_MARKER = {0.025: "^", 0.05: "s", 0.1: "o"}


def load_new() -> tuple[pd.DataFrame, pd.DataFrame]:
    cs, ds = [], []
    for s in SEEDS:
        pc, pd_ = (Path(f"figures/etad_collapse_s{s}.csv"),
                   Path(f"figures/delayed_fixedd_s{s}.csv"))
        if not pc.exists():
            raise SystemExit(f"missing {pc} — run etad_collapse_experiment.py --seed {s}")
        if not pd_.exists():
            raise SystemExit(f"missing {pd_} — run delayed_fixedd_experiment.py --seed {s}")
        cs.append(pd.read_csv(pc).assign(seed=s))
        ds.append(pd.read_csv(pd_).assign(seed=s))
    return pd.concat(cs, ignore_index=True), pd.concat(ds, ignore_index=True)


def prior_d65_late_stats() -> dict:
    """η=0.05, d=6.5 late-window stats from the PRIOR fixed-d round (GPU).
    Same convention as analyze_fixedd.late_stats: late = step >= 1500,
    diverged seed = NaN in the late λmax window (or loss-diverged summary)."""
    per_seed = []
    for s in SEEDS:
        t = pd.read_csv(f"figures/fixedd_traj_s{s}.csv")
        g = t[t.opt == "fixedd_d6.5"]
        late = g[g.step >= LATE_STEP]
        lam = late["lambda_max"].to_numpy(float)
        summ = pd.read_csv(f"figures/fixedd_summary_s{s}.csv")
        loss_div = bool(summ[summ.opt == "fixedd_d6.5"]["diverged"].iloc[0])
        ok = lam[~np.isnan(lam)]
        per_seed.append(dict(
            seed=s, diverged=bool(np.isnan(lam).any() or loss_div),
            late_mean=float(ok.mean()) if ok.size else np.nan))
    st = pd.DataFrame(per_seed)
    surv = st[~st.diverged]
    m = float(surv.late_mean.mean()) if len(surv) else np.nan
    sd = float(surv.late_mean.std(ddof=1)) if len(surv) > 1 else 0.0
    return dict(eta=LR_PRIOR, d=PRIOR_D65, n_seeds=len(st), n_survivors=len(surv),
                late_lmax_mean=m, late_lmax_sd=sd)


def build_collapse_summary(cl: pd.DataFrame) -> pd.DataFrame:
    # round 13+: d grid and source label come from the per-seed CSVs (which
    # now carry backend meta columns); fall back to round-12 semantics for
    # old CSVs without a source column.
    src = str(cl["source"].iloc[0]) if "source" in cl.columns else "cpu_new"
    meta_cols = ("device", "torch_dtype", "torch_version")
    ds = sorted(cl.d.unique())
    rows = []
    for eta in ETAS:
        for d in ds:
            g = cl[(cl.eta == eta) & (cl.d == d)]
            surv = g[(~g.diverged) & (g.late_n_nan == 0)]
            n = len(surv)
            lm = surv.late_mean.to_numpy(float)
            mean = float(lm.mean()) if n else np.nan
            sd = float(lm.std(ddof=1)) if n > 1 else (0.0 if n else np.nan)
            row = dict(
                eta=eta, d=d, n_seeds=len(g), n_survivors=n,
                late_lmax_mean=mean, late_lmax_sd=sd,
                etalam2_mean=eta * mean / 2 if n else np.nan,
                etalam2_sd=eta * sd / 2 if n else np.nan,
                predicted_lameq=2 * d / eta,
                ratio=mean / (2 * d / eta) if n else np.nan,
                source=src)
            for k in meta_cols:
                if k in g.columns:
                    row[k] = g[k].iloc[0]
            if "timestamp" in g.columns:
                row["run_timestamp"] = str(min(g["timestamp"]))
            rows.append(row)
    # prior-round d=6.5 anchor ONLY when 6.5 was not re-measured in this round
    if PRIOR_D65 not in ds:
        p = prior_d65_late_stats()
        rows.append(dict(eta=p["eta"], d=p["d"], n_seeds=p["n_seeds"],
                         n_survivors=p["n_survivors"],
                         late_lmax_mean=p["late_lmax_mean"],
                         late_lmax_sd=p["late_lmax_sd"],
                         etalam2_mean=p["eta"] * p["late_lmax_mean"] / 2,
                         etalam2_sd=p["eta"] * p["late_lmax_sd"] / 2,
                         predicted_lameq=2 * p["d"] / p["eta"],
                         ratio=p["late_lmax_mean"] / (2 * p["d"] / p["eta"]),
                         source="fixedd_prior"))
    return pd.DataFrame(rows)


def build_delayed_summary(dl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        g = dl[dl.seed == s].set_index("arm")
        t_gate = int(g.loc["rankk_k1", "T_gate"])
        rows.append(dict(level="seed", seed=s, arm="rankk_k1 (adaptive)", d=np.nan,
                         T_gate=t_gate, diverged=bool(g.loc["rankk_k1", "diverged"]),
                         survivors=np.nan if g.loc["rankk_k1", "diverged"] else 1,
                         late_lmax=g.loc["rankk_k1", "late_mean"],
                         etalam2=LR_PRIOR * g.loc["rankk_k1", "late_mean"] / 2))
        for d in (2.5, 3.0):
            r = g.loc[f"delayed_d{d:g}"]
            rows.append(dict(level="seed", seed=s, arm=f"delayed_d{d:g}", d=d,
                             T_gate=t_gate, diverged=bool(r.diverged),
                             survivors=np.nan if (r.diverged or r.late_n_nan > 0) else 1,
                             late_lmax=r.late_mean,
                             etalam2=LR_PRIOR * r.late_mean / 2
                             if r.late_mean == r.late_mean else np.nan))
    for arm, d in (("rankk_k1 (adaptive)", np.nan), ("delayed_d2.5", 2.5),
                   ("delayed_d3", 3.0)):
        sub = [r for r in rows if r["arm"] == arm]
        n_surv = int(np.nansum([r["survivors"] for r in sub]))
        lm = np.array([r["late_lmax"] for r in sub], float)
        lm_ok = lm[~np.isnan(lm)] if n_surv else np.array([np.nan])
        rows.append(dict(level="agg", seed=-1, arm=arm, d=d,
                         T_gate=float(np.mean([r["T_gate"] for r in sub])),
                         diverged=bool(n_surv == 0),
                         survivors=n_surv,
                         late_lmax=float(np.mean(lm_ok)),
                         etalam2=LR_PRIOR * float(np.mean(lm_ok))))
    return pd.DataFrame(rows)


REPR_TRAJ = Path("figures/etad_traj_repr_s0.csv")   # panel (c), seed 0
RANK1_ESCAPE = 261.8          # rank-1 damping escape ceiling (paper §4.3)


def _rel_luminance(rgba) -> float:
    """Relative luminance (0..1) of an sRGB color — simple weighted sum of
    the (gamma-encoded) RGB channels. Used to flip heatmap cell text
    black/white for contrast against the cell background (R29: cividis)."""
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fig_unified(summ: pd.DataFrame):
    """Main-text 3-panel figure (R30 restyle, FROZEN 766.872 x 300.092 pt
    page, savefig_fixed — no tight crop; margins are hand-set so nothing
    touches the border). Typography comes from setup(scale=FIG47_SCALE):
    final on-paper sizes tick/legend 7.0, label 7.5, panel tag 8.0 pt."""
    S = FIG47_SCALE
    fig, axes = plt.subplots(
        1, 3, figsize=frozen_figsize("47_etad_collapse"),
        gridspec_kw=dict(width_ratios=[0.94, 1.36, 1.66]))
    # R38 (user): legend moved to the BOTTOM (two stacked rows) and panel (c)
    # trajectories keyed by the legend instead of the in-plot label column.
    # Height re-baselined 300.092 -> 360 pt (user: height may grow), include
    # width 0.83 -> 0.95\linewidth; FIG47_SCALE keeps final fonts locked.
    fig.subplots_adjust(left=0.070, right=0.988, top=0.930, bottom=0.340,
                        wspace=0.365)
    axA, axB, axC = axes

    # ---- (a) collapse: one damping hue, marker shape per η, open = off ---
    lo, hi = 1.0, 7.2
    axA.plot([lo, hi], [lo, hi], ls="--", color=C["theory"], lw=ST["theory"],
             zorder=1)
    # R38 (user): η dimension now carries a MONOTONE SHADE ramp as well
    # (light η=0.025 -> dark η=0.1, fig-33 _mix_white style) on top of the
    # marker-shape channel; on- vs off-identity stays filled vs open.
    ETA_SHADE = {0.025: 0.38, 0.05: 0.16, 0.1: 0.0}
    labels = []   # (d, y, text, anomalous) — deduped before annotating
    for eta in ETAS:
        m = ETA_MARKER[eta]
        col = _mix_white(C["damping"], ETA_SHADE[eta])
        sub = summ[(summ.eta == eta) & (summ.source != "fixedd_prior")
                   & (summ.n_survivors > 0)]
        if not len(sub):
            continue
        on = sub[sub.ratio >= 0.9]
        off = sub[sub.ratio < 0.9]
        if len(on):
            axA.errorbar(on.d, on.etalam2_mean, yerr=on.etalam2_sd,
                         marker=m, ms=10.5, lw=0, elinewidth=ST["errbar"],
                         capsize=ST["capsize"], mfc=col,
                         mec=col, mew=1.6, ecolor=col,
                         zorder=3)
        if len(off):
            axA.errorbar(off.d, off.etalam2_mean, yerr=off.etalam2_sd,
                         marker=m, ms=11.0, lw=0, elinewidth=ST["errbar"],
                         capsize=ST["capsize"], mfc="none", mec=col,
                         mew=1.9, ecolor=col, zorder=3)
        # R30: the repeated 5/5 counts are muted background info; only the
        # sub-full-survival cells (2/5, 3/5) are emphasized. Near-degenerate
        # cells (same d, |Δy| < 0.22) share one slot: the anomalous count
        # wins, identical 5/5 twins are drawn once.
        for _, r in pd.concat([on, off]).iterrows():
            labels.append((float(r.d), float(r.etalam2_mean),
                           f"{int(r.n_survivors)}/5",
                           bool(r.n_survivors < 5)))
    labels.sort(key=lambda t: (not t[3], t[0], t[1]))   # anomalous first
    kept = []
    for d, y, txt, anom in labels:
        if any(abs(d - d0) < 0.55 and abs(y - y0) < 0.22
               for d0, y0, _, _ in kept):
            continue
        kept.append((d, y, txt, anom))
    axA.set_xlabel(r"fixed divisor $d$", labelpad=7)
    axA.set_ylabel(r"$\eta\,\bar{\lambda}_{\mathrm{late}}/2$")
    axA.set_xlim(lo, hi)
    axA.set_ylim(lo, hi)
    # R38 (user): drop aspect="equal" so panel (a) matches (b)/(c) height —
    # the identity line stays the y=x reference (value equality); its slope
    # is no longer 45 deg on paper, which the caption wording survives.
    axA.set_aspect("auto")
    panel_label(axA, "a", "Scaled collapse")

    # figure-level key, MOVED TO THE BOTTOM (R38) as two stacked rows:
    #   row 1 (panel a): the three η marker shapes + off-identity + identity
    #   row 2 (panel c): the three trajectory linestyles with regime+params
    # R38 (user): the legend must match what is actually drawn. η=0.1 is
    # 0/5 in EVERY cell (panel b, bottom... top row) so no η=0.1 marker
    # exists in panel (a) — its circle entry is dropped. The only
    # off-identity marker drawn is the OPEN η=0.025 TRIANGLE, so the
    # "survived, off identity" sample is that open light triangle (was an
    # open circle nothing pointed to).
    key_handles = [
        Line2D([], [], marker=ETA_MARKER[eta], ls="none",
               mfc=_mix_white(C["damping"], ETA_SHADE[eta]),
               mec=_mix_white(C["damping"], ETA_SHADE[eta]),
               mew=1.6, ms=10.5,
               label=rf"$\eta={eta:g}$")
        for eta in (0.025, 0.05)
    ] + [
        Line2D([], [], marker="^", ls="none", mfc="none",
               mec=_mix_white(C["damping"], ETA_SHADE[0.025]),
               mew=1.9, ms=11.0, label="survived, off identity"),
        Line2D([], [], ls="--", color=C["theory"], lw=ST["theory"],
               label=r"identity $y=x$"),
    ]
    # (user round: row 1 dropped closer to row 2 — anchor 0.115→0.082 trims
    # the inter-row ink gap from ~18.7pt to ~7.9pt at the 324pt canvas,
    # matching the 45 legend rhythm; row 2 anchor untouched.)
    fig.legend(handles=key_handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.082), handlelength=1.3,
               columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0)

    # ---- (b) survival heatmap: all 18 cells (cividis kept) ---------------
    cmap = plt.get_cmap("cividis")
    new = summ[summ.source != "fixedd_prior"]
    dvals = sorted(new.d.unique())
    pos = {d: i for i, d in enumerate(dvals)}
    xs, ys, fracs = [], [], []
    for yi, eta in enumerate(ETAS):
        sub = new[new.eta == eta]
        for _, r in sub.iterrows():
            frac = r.n_survivors / r.n_seeds
            xs.append(pos[r.d])
            ys.append(yi)
            fracs.append(frac)
            axB.annotate(f"{int(r.n_survivors)}/5", (pos[r.d], yi),
                         ha="center", va="center", fontsize=8.0 * S,
                         fontweight="semibold", zorder=4,
                         color="white" if _rel_luminance(cmap(frac)) < 0.5
                         else "black")
    coll = axB.scatter(xs, ys, s=800, c=fracs, vmin=0, vmax=1, cmap=cmap,
                       marker="s", edgecolors=C["theory"], linewidths=1.1,
                       zorder=3)
    axB.set_xticks(range(len(dvals)))
    axB.set_xticklabels([f"{d:g}" for d in dvals])
    axB.set_yticks(range(len(ETAS)))
    # R38 (user): "η" horizontal (not rotated); tick values stacked
    # vertically. The ylabel itself is pinned AFTER the colorbar call
    # (colorbar re-layouts axB and would reset the label position).
    axB.set_yticklabels([f"{eta:g}" for eta in ETAS])
    axB.set_xlim(-0.6, len(dvals) - 0.4)
    axB.set_ylim(-0.6, len(ETAS) - 0.4)
    axB.set_xlabel(r"fixed divisor $d$", labelpad=7)
    # R30: colorbar narrow + light (thin bar, sparse ticks; no text label —
    # the panel title "Survival heatmap" + caption carry the wording, and a
    # rotated label would crowd panel (c)'s tick zone)
    cbar = fig.colorbar(coll, ax=axB, ticks=[0, 0.5, 1], pad=0.03,
                        fraction=0.05)
    cbar.outline.set_linewidth(0.6 * S)
    cbar.ax.tick_params(width=0.7 * S, length=2.5 * S)
    panel_label(axB, "b", "Survival heatmap")
    # η label: horizontal, pinned left of the tick column (after the
    # colorbar re-layout — see note above).
    axB.set_ylabel(r"$\eta$", rotation=0, va="center", labelpad=26)
    axB.yaxis.set_label_coords(-0.30, 0.5)

    # ---- (c) representative trajectories (seed 0) ------------------------
    # R38 (user): the three linestyles are keyed by the BOTTOM legend row
    # (regime + params in the legend) — the old in-plot right-edge label
    # column (param + regime word per line) is gone. Predicted levels stay
    # DASHED and the escape ceiling DOTTED (caption coupling), as thin
    # theory references drawn over the DATA SPAN ONLY.
    cells = [
        # (eta, d, ls, marker, regime word, color) — R38 (user): three
        # DISTINCT hues (fig-33 style: one color per series reads at a
        # glance). R39 grayscale-redundancy lane (fixplan_r15 §6): each
        # regime ALSO carries a distinct linestyle + marker shape so panel
        # (c) stays parseable in grayscale print — saturation solid ->
        # dash-dot per the fixplan spec. Blue anchors the well-behaved
        # on-orbit arm; teal / rose mark the intermittent / saturation arms.
        (0.025, 6.5, "-.", "s", "saturation", C["amplification"]),
        (0.025, 2.5, "--", "^", "intermittent", C["sam"]),
        (0.05, 1.5, "-", "o", "on orbit", C["damping"]),
    ]
    if REPR_TRAJ.exists():
        tr = pd.read_csv(REPR_TRAJ)
        x_end = float(tr.step.max())
        traj_key = []
        for eta, d, ls, mkr, word, col in cells:
            g = tr[(tr.eta == eta) & (tr.d == d)].sort_values("step")
            # R39: markers on the three main trajectories only (theory
            # references below stay marker-free); markevery keeps ~10
            # markers per 30-point trajectory.
            axC.plot(g.step, g.lambda_max, color=col, ls=ls, marker=mkr,
                     ms=ST["marker"], markevery=max(1, len(g) // 10),
                     lw=ST["primary"], zorder=3)
            lev = 2 * d / eta
            axC.plot([0, x_end], [lev, lev], ls="--", color=C["theory_light"],
                     lw=ST["theory"], zorder=2)
            traj_key.append(Line2D(
                [], [], color=col, ls=ls, marker=mkr, ms=ST["marker"],
                lw=ST["primary"],
                label=rf"{word} ($\eta={str(eta).lstrip('0')}$, "
                      rf"$d={d:g}$)"))
        axC.plot([0, x_end], [RANK1_ESCAPE, RANK1_ESCAPE], ls=":",
                 color=C["theory"], lw=ST["theory"], zorder=2)
        # R38 (user): the 261.8 tag lives at the RIGHT tip of the ceiling
        # line, near-black (book text spec #1A1D21) and hugging the dotted
        # line (+1.2 pt) — the old mid-gap spot at x=60 read as floating
        # between the 2d/η=520 dashed line above and the ceiling below.
        axC.annotate(r"$261.8$", (x_end, RANK1_ESCAPE),
                     textcoords="offset points", xytext=(0, 1.2),
                     ha="right", va="bottom",
                     fontsize=8.0 * S, color="#1A1D21", zorder=4)
        axC.set_yscale("log")
        axC.set_xlim(0, x_end * 1.02)
        axC.set_ylim(0.3, 850)
        axC.set_xlabel(r"step", labelpad=7)
        panel_label(axC, "c", r"$\lambda_{\mathrm{raw}}(t)$ trajectories")
        # row 2 of the bottom key: the three trajectory linestyles (R38)
        fig.legend(handles=traj_key, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 0.012),
                   handlelength=1.6, columnspacing=1.1, handletextpad=0.5,
                   borderaxespad=0.0)
    else:
        axC.text(0.5, 0.5, f"missing {REPR_TRAJ}", ha="center",
                 transform=axC.transAxes)

    # ---- panel (a) count labels, placed LAST with real geometry ----------
    # Everything that can move axA (aspect="equal" resizes the axes box,
    # panel_label/title, colorbar steal on axB) is already in place, so the
    # renderer-measured boxes match the saved page. Rule: anomalous counts
    # (2/5, 3/5) always render (both offsets tried); a MUTED 5/5 renders
    # only where it fits without colliding with another label, the title,
    # or the tick labels — the dense d=1.25–2.5 cluster has no clean slot
    # and those full-survival counts stay available in panel (b) + caption.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _obstacles(exclude):
        boxes = []
        for t in list(axA.texts) + list(axA.get_xticklabels()) \
                + list(axA.get_yticklabels()):
            if t is exclude:
                continue
            try:
                boxes.append(t.get_window_extent(renderer))
            except Exception:
                pass
        return boxes

    for d, y, txt, anom in kept:
        cands = ([(0, -15, "top"), (0, 11, "bottom")] if y > 5.5
                 else [(0, 11, "bottom"), (0, -15, "top")])
        placed = None
        for dx, dy, va in cands:
            ann = axA.annotate(txt, (d, y), textcoords="offset points",
                               xytext=(dx, dy), ha="center", va=va,
                               fontsize=8.0 * S,
                               fontweight="semibold" if anom else "normal",
                               color=C["theory"] if anom else C["theory_light"],
                               zorder=4)
            fig.canvas.draw()
            bb = ann.get_window_extent(renderer).padded(2.0)
            axbox = Bbox(axA.get_window_extent(renderer).get_points())
            inside = (axbox.contains(bb.x0, bb.y0)
                      and axbox.contains(bb.x1, bb.y1))
            if not inside or any(bb.overlaps(o) for o in _obstacles(ann)):
                ann.remove()
                continue
            placed = ann
            break
        if placed is None and anom:
            # anomalous counts must render even if both offsets clash
            dx, dy, va = cands[0]
            axA.annotate(txt, (d, y), textcoords="offset points",
                         xytext=(dx, dy), ha="center", va=va,
                         fontsize=8.0 * S, fontweight="semibold",
                         color=C["theory"], zorder=4)
    return fig


def fig_phase(summ: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    # R29 typography: sized for the 0.90\linewidth appendix inclusion
    # (~356pt final). NOTE: shortening the title removed the old
    # width-driving element, so the tight bbox collapsed to W≈513pt and
    # scale rose to ~0.69 — source pts below are computed against that
    # (verified with /tmp/opencode/round29/qa_fig.py). RdYlGn -> cividis
    # (colorblind-safe sequential).
    # R30 apply-only restyle: strokes / colors / spine guards / shortened
    # title. Layout, canvas, annotation positions and the tight save are
    # deliberately UNCHANGED (this figure is not footprint-frozen).
    TICK, LAB, CELL, ANN, TITLE = 11.23, 11.23, 11.23, 11.23, 11.93
    for sp in ax.spines.values():   # -> ~0.75pt final (target 0.6-0.8)
        sp.set_linewidth(1.08)
    # setup(scale=FIG47_SCALE) raises the global rcParams tick/spine widths;
    # pin this figure back to its calibrated unscaled strokes.
    ax.tick_params(width=0.7, length=2.5)
    cmap = plt.get_cmap("cividis")
    # any non-prior row is a current-round cell (source label changed from
    # "cpu_new" to e.g. "cpu_round13" once --source was added)
    new = summ[summ.source != "fixedd_prior"]
    # prior-round d=6.5 point on the η=0.05 row (diamond, same color scale) —
    # only present when 6.5 was NOT re-measured in the current round
    p = summ[summ.source == "fixedd_prior"]
    # R28: x is CATEGORICAL (equal spacing per d value) — the real-d axis
    # crowded the 1.25..3 ticks while stretching the range for d=6.5.
    dvals = sorted(new.d.unique())
    if len(p) and PRIOR_D65 not in dvals:
        dvals.append(PRIOR_D65)
    pos = {d: i for i, d in enumerate(dvals)}
    # Bottom ηλ̄/2 value labels are drawn for FOUR selected cells only
    # (2026-09-24 overlap fix): the fully-surviving left-side cluster
    # d=1.25..2.5 used to get one label per cell and they collided. The
    # four kept cells carry the regime story: on-orbit (η=.05, d=2),
    # saturation (d=6.5 both η), intermittent collapse (η=.025, d=2.5).
    ETALAM2_CELLS = {(0.05, 2.0), (0.05, 6.5), (0.025, 2.5), (0.025, 6.5)}
    xs, ys, fracs = [], [], []
    for yi, eta in enumerate(ETAS):
        sub = new[new.eta == eta]
        for _, r in sub.iterrows():
            frac = r.n_survivors / r.n_seeds
            xs.append(pos[r.d])
            ys.append(yi)
            fracs.append(frac)
            # R29: counts moved INTO the discs (10pt → 13pt above-cell labels
            # no longer fit the inter-row band next to the ηλ̄/2 labels);
            # text color flips black/white by cell luminance (cividis).
            ax.annotate(f"{int(r.n_survivors)}/5", (pos[r.d], yi),
                        ha="center", va="center", fontsize=CELL,
                        fontweight="bold", zorder=4,
                        color="white" if _rel_luminance(cmap(frac)) < 0.5
                        else "black")
            if r.n_survivors > 0 and (eta, float(r.d)) in ETALAM2_CELLS:
                d65 = float(r.d) == 6.5  # near the right edge: right-align
                # R28: -14 → -19; R29: -19 → -28 (13pt glyphs need the
                # extra clearance below the disc).
                ax.annotate(rf"$\eta\bar\lambda/2={r.etalam2_mean:.2f}$",
                            (pos[r.d], yi), textcoords="offset points",
                            xytext=(-12 if d65 else 0, -28),
                            ha="right" if d65 else "center", fontsize=ANN,
                            color=C["theory"], zorder=4)
    coll = ax.scatter(xs, ys, s=560, c=fracs, vmin=0, vmax=1, cmap=cmap,
                      marker="o", edgecolors=C["theory"], linewidths=1.05,
                      zorder=3)
    if len(p):
        pr = p.iloc[0]
        yi = ETAS.index(pr.eta)
        ax.scatter([pos[pr.d]], [yi], s=560, c=[pr.n_survivors / pr.n_seeds],
                   vmin=0, vmax=1, cmap=cmap, marker="D",
                   edgecolors=C["theory"], linewidths=1.05, zorder=3)
        ax.annotate(f"{int(pr.n_survivors)}/5", (pos[pr.d], yi),
                    ha="center", va="center", fontsize=CELL,
                    fontweight="bold", zorder=4,
                    color="white" if _rel_luminance(
                        cmap(pr.n_survivors / pr.n_seeds)) < 0.5
                    else "black")
        ax.annotate(rf"$\eta\bar\lambda/2={pr.etalam2_mean:.2f}$" + "\n(prior round)",
                    (pos[pr.d], yi), textcoords="offset points", xytext=(0, -42),
                    ha="center", fontsize=ANN, color=C["theory"])
        # non-monotone callout on the η=0.05 row: d=3 → 0/5 but d=6.5 → 3/5.
        # R28 remap: absolute-d coords (6.28 / 3.22 / midpoint 4.75) became
        # position-space offsets — tail just right of d=3, head just left of
        # d=6.5, caption text centered above the midpoint.
        x_head = pos[PRIOR_D65] - 0.22
        x_tail = pos[3.0] + 0.18
        ax.annotate("", xy=(x_head, yi + 0.06), xytext=(x_tail, yi + 0.06),
                    arrowprops=dict(arrowstyle="->", color=C["theory"],
                                    lw=1.1,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text((x_head + x_tail) / 2, yi + 0.42,
                "non-monotone in $d$:\n$d{=}2.5,3$ diverge, $d{=}6.5$ survives",
                ha="center", fontsize=ANN, color=C["theory"])
    ax.set_xticks(range(len(dvals)))
    ax.set_xticklabels([f"{d:g}" + ("\n(prior)" if len(p) and d == PRIOR_D65
                                     and d not in set(new.d.unique()) else "")
                        for d in dvals], fontsize=TICK)
    ax.set_yticks(range(len(ETAS)))
    ax.set_yticklabels([rf"$\eta={eta:g}$" for eta in ETAS], fontsize=TICK)
    ax.tick_params(labelsize=TICK)
    ax.set_xlim(-0.7, len(dvals) - 0.3)
    ax.set_ylim(-0.75, len(ETAS) - 0.25 + 0.55)
    ax.set_xlabel(r"fixed divisor $d$", fontsize=LAB)
    # R29: explanatory clause moved to the LaTeX caption; title shortened.
    # R30: compressed further ("stability" carried by the axis content).
    ax.set_title("Fixed-$d$ phase diagram", fontsize=TITLE)
    cbar = fig.colorbar(coll, ax=ax, ticks=[0, 0.5, 1], pad=0.02)
    cbar.set_label("survival fraction (seeds / 5)", fontsize=ANN)
    cbar.ax.tick_params(labelsize=TICK, width=0.7, length=2.5)
    cbar.outline.set_linewidth(0.75)
    return fig


def _save(fig, name: str, pad: float):
    """Local tight save for the NON-frozen appendix figure (48). The frozen
    main-text figure (47) uses figure_style.savefig_fixed instead."""
    from steos.analysis.figure_style import FIG_DIR
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out, pad_inches=pad)
    plt.close(fig)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true",
                    help="re-aggregate the summary CSVs only; do NOT redraw "
                         "47/48 (paper figures are a separate lane)")
    args = ap.parse_args()

    # R30: typography/strokes from the shared visual system; the R29 local
    # font overrides are gone. scale maps source pt -> final on-paper pt
    # (report("figures/47_etad_collapse.pdf", "47_etad_collapse") prints it).
    setup(scale=FIG47_SCALE)

    # R30 data-layer guard (orchestrator-authorized minimal IO guard): the
    # published summary CSVs are read back when present and never rewritten;
    # they are computed + written only when missing. The analysis logic
    # itself is unchanged.
    summ_path = Path("figures/etad_collapse_summary.csv")
    dsum_path = Path("figures/delayed_fixedd_summary.csv")
    if summ_path.exists() and dsum_path.exists():
        summ = pd.read_csv(summ_path)
        dsum = pd.read_csv(dsum_path)
        print(f"[guard] {summ_path.name} + {dsum_path.name} exist — "
              "loaded read-only, not rewritten")
    else:
        cl, dl = load_new()
        summ = build_collapse_summary(cl)
        dsum = build_delayed_summary(dl)
        summ.to_csv(summ_path, index=False)
        dsum.to_csv(dsum_path, index=False)

    pd.set_option("display.width", 240)
    print(f"=== η×d collapse summary (late window step >= {LATE_STEP}; "
          "survivor = not diverged & no late NaN) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    print("\n=== VALIDATION (a): η=0.05 anchors vs prior round "
          "(figures/fixedd_summary_agg.csv, GPU) ===")
    for d, prior in PRIOR_ANCHORS.items():
        r = summ[(summ.eta == 0.05) & (summ.d == d)].iloc[0]
        rel = abs(r.late_lmax_mean - prior) / prior * 100 if np.isfinite(
            r.late_lmax_mean) else float("nan")
        print(f"  d={d:<4g}: new CPU late λ̄ = {r.late_lmax_mean:7.2f}  "
              f"prior = {prior:7.2f}  |Δ|/prior = {rel:4.1f}%  "
              f"(pred 2d/η = {2*d/0.05:g})")

    print("\n=== delayed-onset fixed-d control (VALIDATION (b)) ===")
    print(dsum.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    outs = []
    if not args.no_figures:
        f1 = fig_unified(summ)
        # R39 atomic regeneration (another session may be compiling the
        # paper): save to <name>.tmp.pdf first, mv over the live PDF only
        # after a successful save.
        tmp1 = savefig_fixed(f1, "47_etad_collapse.tmp")
        out1 = tmp1.with_name("47_etad_collapse.pdf")
        tmp1.replace(out1)
        outs.append(out1)
        f2 = fig_phase(summ)
        outs.append(_save(f2, "48_fixedd_phase", pad=0.085))
    print(f"\n=> figures/etad_collapse_summary.csv + figures/delayed_fixedd_summary.csv"
          + ("".join(f"\n=> {o} ({o.stat().st_size} bytes)" for o in outs)))


if __name__ == "__main__":
    main()
