"""CIFAR-100 / ResNet-18 control-experiment analysis: SGD vs rank-1 plant vs SAM.

Headline figure for reviewer concern #1 ("experimental scale"): does the
"predictors-are-not-actuators" split (the paper's central claim) reproduce on a
TRUE ResNet-18 (11.2M params) on CIFAR-100, instead of MLP / toy-ResNet?

Three-way comparison:
  - SGD baseline: EoS — lambda_max self-stabilizes at the heavy-ball stability
    boundary 2(1+mu)/eta, never durably below.
  - rank-1 plant: Claim-1 actuator — should FAIL to push lambda_max below the
    heavy-ball boundary (the self-stabilization cap blocks it).
  - SAM:         global action — should escape below the heavy-ball boundary,
    because its perturbation is global, not rank-1.

IMPORTANT (methodology): these runs use momentum SGD (mu=0.9). The correct
linear-stability / EoS reference is therefore the HEAVY-BALL boundary
2(1+mu)/eta (=38 for eta=0.1, mu=0.9), NOT the plain GD boundary 2/eta (=20).
Judging "escape" against 2/eta is wrong for momentum runs and makes the verdict
meaningless; we use 2(1+mu)/eta as the operative threshold throughout.

Reads:  runs/cifar100_r18_{sgd,rank1,sam}_detarpack_s*/metrics.csv    (canonical: deterministic CUDA + ARPACK true-lambda_max, every-100, n=5)
        runs/cifar100_r18_{sgd,rank1,sam}_detarpack_d20_s*/metrics.csv (dense: same protocol, outcome-every-20 -> 251 pts/run; trajectories only)
        runs/cifar100_r18_rankk{2,5}_s*/metrics.csv                  (optional; original power-iter protocol)
Writes: figures/30_cifar100_r18_control.pdf
        figures/cifar100_control_summary.csv

TWO data sources (do not conflate):
  * TRAJECTORIES, panels (a)/(b): the dense _detarpack_d20 runs. Identical
    protocol/seeds/deterministic CUDA (train_loss at overlapping steps is
    bit-identical), but outcome-every=20 gives 251 points per curve instead
    of 51 -> smooth figure trajectories.
  * STATISTICS, panel (c) + summary CSV + console numbers: the canonical
    _detarpack runs (every-100). This is the protocol of record; the paper's
    late-window statistics (29.21±1.90 etc.) must NOT drift with the
    measurement sampling density. The dense runs' lambda_max differs from
    canonical at overlapping steps by <=~1.2% rel (max observed) because the
    train-mode tr_loss forward inside the measurement block nudges BN
    running stats once per measurement — a measurement-side caveat of
    measurement frequency, not a training-trajectory difference.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_MARKERS,
    ST,
    THEORY,
    panel_label,
    report,
    savefig,
    setup,
)

# Source canvas is included at 0.95\linewidth and shrunk accordingly; the
# scale factor keeps the FINAL on-paper fonts at the R30 floors (see the
# report() output at the end of the run).
setup(scale=1.2741)

RUNS = Path("runs")
FIG = Path("figures")

# window definition for "late training" statistics
LATE_STEP = 2000

# map opt -> (display name, color, marker); colors + markers from the global
# METHOD_COLORS / METHOD_MARKERS registries (same method = same hue and
# marker in every paper figure). R39 (fixplan §5.11): display names are the
# caption's four-arm vocabulary — the legacy clamped sensor arm is now
# labeled "legacy sensor" to distinguish it from the corrected
# ARPACK-sensed arm added in R39.
OPTS = [
    ("sgd",   "SGD",                    METHOD_COLORS["sgd"],   METHOD_MARKERS["sgd"]),
    ("rank1", "rank-1 (legacy sensor)", METHOD_COLORS["rank1"], METHOD_MARKERS["rank1"]),
    ("sam",   "SAM",                    METHOD_COLORS["sam"],   METHOD_MARKERS["sam"]),
]

# R39 (fixplan §5.11/§6): the corrected ARPACK-sensed rank-1 arm (the D.14
# sensor fix, drawn into the figure for the first time). Source of record =
# runs/cifar100_r18_rankkx_d1_s* — manifest opt=rankkx / edit=damp / k=1 /
# plant_N=20, metrics.csv carries the ARPACK outcome λref (lambda_max) and
# eval_acc. This is the ONLY correct source: NOT runs/cifar100_r18_rank1_
# noclamp_s* (outcome_arpack:false, the Fig-7 family) and NOT runs/cifar_ws_s*
# (power-iter warm-start). Deliberately kept OUTSIDE OPTS so the frozen
# figures/cifar100_control_summary.csv (3 canonical arms) can never absorb
# it; its statistics go to the separate cifar100_control_rankkx_summary.csv.
CORR_OPT = "rankkx_d1"
CORR_NAME = "rank-1 (ARPACK-sensed)"
CORR_COLOR = METHOD_COLORS["rankkx_d1"]   # damping blue — same family hue as
CORR_MARKER = "s"                         # the legacy arm; separated by
                                          # linestyle (-. vs :) + marker (s vs x)

# direct-label / threshold-tag size (pt on the source canvas)
FS_TAG = ST["note"]


def smooth32(mean_traj, sigma=12.0):
    """Fig-32 parity display smoothing (user round: 图30 平滑改为和32图
    类似的方式和程度): Gaussian filter on log10 of the cross-seed mean
    (σ=12 samples ≈ 240 steps on the dense every-20 grid, reflect edges);
    the first 3 samples (steps 0–40, where the init spike lives off-frame)
    keep the raw mean, crossfaded log-linearly into the smoothed curve over
    samples 2–12 (steps 40–240) — verbatim the plot_bridge_clampfree.py
    recipe. DISPLAY-LAYER ONLY: panel (c) statistics, the summary CSV and
    every console number come from the canonical CSVs and never see this.
    R39: `sigma` is exposed so the corrected arm's canonical every-100
    trajectories (no dense _d20 variant exists) can keep the same PHYSICAL
    smoothing width (240 steps = 2.4 samples at every-100)."""
    lg = np.log10(mean_traj.to_numpy())
    sm_lg = gaussian_filter1d(lg, sigma=sigma, mode="reflect")
    w = np.clip((np.arange(len(lg)) - 2) / (12 - 2), 0, 1)
    return pd.Series(10 ** ((1 - w) * lg + w * sm_lg), index=mean_traj.index)


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def ci95(x):
    """95% CI half-width of a sample via Student-t (falls back to 1.96*z).

    Returns 0.0 for n<2 (single value: no uncertainty to report).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return 0.0
    sd = x.std(ddof=1)
    try:
        from scipy.stats import t  # graceful fallback if scipy missing
        tcrit = t.ppf(0.975, n - 1)
    except Exception:
        tcrit = 1.96
    return float(tcrit * sd / np.sqrt(n))


def fmt_ci(mean, ci):
    """Format `mean ± ci`; for a degenerate single-sample CI show the value only."""
    if ci == 0.0 or np.isnan(ci):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {ci:.2f}"


# --------------------------------------------------------------------------- #
# discovery / loading
# --------------------------------------------------------------------------- #
def discover(opt: str, suffix: str = "_detarpack") -> list[tuple[int, Path, Path]]:
    """Return [(seed, metrics_csv, manifest_json)] for every non-smoke run of `opt`.

    suffix="_detarpack" reads the canonical deterministic+ARPACK three-way runs
    (cifar100_r18_{opt}_detarpack_s{seed}). Pass suffix="" to read the original
    non-deterministic power-iter runs (used by rankk_check for k=1 consistency
    with the k=2,5 sweeps, which were run on that older protocol).

    R40 exact-match fix (same rule as analyze_rankkx_interp.late_stats): the
    old trailing-'*' glob also swallowed sibling families — the canonical SGD
    prefix matched the '_spec' spectrum-diagnostic reruns (same seeds, same
    trajectory plus an extra measurement), double-counting seeds 0/1/2 and
    freezing sgd n=8 (28.98±1.26) into the summary CSV instead of the
    canonical n=5 (29.21±1.90). Directory names must now be exactly
    prefix + digits; '_spec_'/'_d20_' siblings can never match.
    """
    prefix = f"cifar100_r18_{opt}{suffix}_s"
    pat = re.compile(re.escape(prefix) + r"(\d+)")
    out = []
    for d in sorted(RUNS.iterdir()):
        m = pat.fullmatch(d.name)
        if m is None or not d.is_dir() or "smoke" in d.name:
            continue
        csv = d / "metrics.csv"
        if not csv.exists():
            continue
        out.append((int(m.group(1)), csv, d / "manifest.json"))
    return out


def read_hyperparams(records):
    """Read (lr, momentum) from manifests; warn if inconsistent across runs."""
    lrs, mus = [], []
    for _, _, mf in records:
        try:
            import json
            j = json.loads(mf.read_text())
            if j.get("lr") is not None:
                lrs.append(float(j["lr"]))
            if j.get("momentum") is not None:
                mus.append(float(j["momentum"]))
        except Exception:
            pass
    if not lrs:
        raise RuntimeError("no lr found in any manifest")
    lr = float(np.median(lrs))
    mu = float(np.median(mus)) if mus else 0.0
    if len(set(lrs)) > 1 or len(set(mus)) > 1:
        print(f"[warn] inconsistent lr/momentum across runs: lr={set(lrs)} mu={set(mus)}; "
              f"using median lr={lr}, mu={mu}")
    return lr, mu


def per_seed_stats(df: pd.DataFrame, hb: float) -> dict:
    """Late-window statistics for a single seed's metrics DataFrame."""
    late = df[df.step >= LATE_STEP]
    lam = late.lambda_max.to_numpy(dtype=float)
    return dict(
        n_late=len(lam),
        lambda_late_mean=float(np.mean(lam)) if lam.size else float("nan"),
        lambda_late_min=float(np.min(lam)) if lam.size else float("nan"),
        frac_below_HB=float(np.mean(lam < hb)) if lam.size else float("nan"),
        final_acc=float(df.eval_acc.iloc[-1]),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    FIG.mkdir(exist_ok=True)

    # ---- discover runs per optimizer ----------------------------------------
    # TWO datasets, intentionally kept separate (see module docstring):
    #   per_opt      canonical _detarpack  -> ALL statistics (panel (c), summary
    #                                     CSV, console numbers; protocol of record)
    #   per_opt_d20  dense _detarpack_d20  -> trajectory panels (a)/(b) only
    per_opt = {opt: discover(opt) for opt, *_ in OPTS}
    per_opt_d20 = {opt: discover(opt, suffix="_detarpack_d20") for opt, *_ in OPTS}
    present = [(opt, name, c, m) for (opt, name, c, m) in OPTS if per_opt[opt]]
    if not present:
        print("no runs found; expected runs/cifar100_r18_{sgd,rank1,sam}_s*/metrics.csv")
        return

    # R39: corrected ARPACK-sensed rank-1 arm (see CORR_OPT note above).
    corr_records = discover(CORR_OPT, suffix="")
    have_corr = bool(corr_records)
    if not have_corr:
        print(f"[warn] no {CORR_OPT} runs found (runs/cifar100_r18_{CORR_OPT}_s*) — "
              "the corrected arm will be absent from the figure")
    else:
        print(f"[r39] corrected arm: {len(corr_records)} seeds "
              f"{[s for s, _c, _m in corr_records]} from runs/cifar100_r18_{CORR_OPT}_s*")

    # display order for the trajectory panels + panel (c): SGD, legacy
    # rank-1, corrected ARPACK-sensed rank-1, SAM
    arms = []
    for arm in present:
        if arm[0] == "sam" and have_corr:
            arms.append((CORR_OPT, CORR_NAME, CORR_COLOR, CORR_MARKER))
        arms.append(arm)

    # ---- thresholds from manifests (do NOT hardcode) ------------------------
    all_records = [r for recs in per_opt.values() for r in recs] + corr_records
    lr, mu = read_hyperparams(all_records)
    GD_THRESH = 2.0 / lr                                   # plain GD:  2/eta       (=20)
    HB_THRESH = 2.0 * (1.0 + mu) / lr                      # OPERATIVE: 2(1+mu)/eta (=38)

    print("=== CIFAR-100 / ResNet-18 control — thresholds ===")
    print(f"  lr (eta)            = {lr}")
    print(f"  momentum (mu)       = {mu}")
    print(f"  2/eta  (GD)         = {GD_THRESH:g}")
    print(f"  2(1+mu)/eta (HB)    = {HB_THRESH:g}   <-- OPERATIVE stability boundary "
          f"(momentum-SGD runs; the EoS reference for these runs)")

    # ---- per-seed stats -----------------------------------------------------
    # seed_data[opt] = list of (seed, df, stats)   — CANONICAL _detarpack:
    # feeds panel (c), the summary CSV and every console number below.
    seed_data: dict[str, list] = {}
    for opt, _, _, _ in present:
        rows = []
        for seed, csv, _mf in per_opt[opt]:
            df = pd.read_csv(csv).sort_values("step").reset_index(drop=True)
            rows.append((seed, df, per_seed_stats(df, HB_THRESH)))
        seed_data[opt] = rows

    # seed_traj[opt] = list of (seed, df)          — DENSE _detarpack_d20:
    # 251-point trajectories for panels (a)/(b) ONLY (no stats computed here;
    # statistics must not drift with sampling density — see module docstring).
    # Falls back to canonical if the dense runs are absent on this machine.
    seed_traj: dict[str, list] = {}
    for opt, _, _, _ in present:
        recs = per_opt_d20[opt] or per_opt[opt]
        if per_opt[opt] and not per_opt_d20[opt]:
            print(f"[warn] no _detarpack_d20 runs for opt={opt}; falling back to "
                  f"canonical every-100 trajectories in panels (a)/(b)")
        seed_traj[opt] = [(seed, pd.read_csv(csv).sort_values("step").reset_index(drop=True))
                          for seed, csv, _mf in recs]

    # R39: corrected-arm data. seed-level stats from the CANONICAL every-100
    # runs (same per-seed late-window recipe as the other arms — never
    # pooled steps); trajectories fall back to the same canonical grid (no
    # dense _d20 variant exists for this arm), with smooth32's sigma scaled
    # to keep the physical smoothing width (see smooth32 docstring).
    corr_seed_data: list = []
    corr_seed_traj: list = []
    corr_traj_dense = False
    corr_agg: dict = {}  # filled iff have_corr; CORR_OPT enters `arms` iff have_corr
    if have_corr:
        corr_d20 = discover(CORR_OPT, suffix="_detarpack_d20")
        for seed, csv, _mf in corr_records:
            df = pd.read_csv(csv).sort_values("step").reset_index(drop=True)
            corr_seed_data.append((seed, df, per_seed_stats(df, HB_THRESH)))
        recs = corr_d20 or corr_records
        corr_traj_dense = bool(corr_d20)
        if not corr_d20:
            print(f"[note] no dense _d20 runs for opt={CORR_OPT}; panels (a)/(b) "
                  "use the canonical every-100 trajectories (display only)")
        corr_seed_traj = [(seed, pd.read_csv(csv).sort_values("step").reset_index(drop=True))
                          for seed, csv, _mf in recs]
        lam_means = np.array([s["lambda_late_mean"] for _s, _d, s in corr_seed_data])
        accs = np.array([s["final_acc"] for _s, _d, s in corr_seed_data])
        corr_agg = dict(
            n_seeds=len(lam_means),
            lambda_late_mean=float(np.mean(lam_means)),
            lambda_late_ci=ci95(lam_means),
            lambda_late_min_mean=float(np.mean(
                [s["lambda_late_min"] for _s, _d, s in corr_seed_data])),
            frac_below_HB_mean=float(np.mean(
                [s["frac_below_HB"] for _s, _d, s in corr_seed_data])),
            frac_below_HB_ci=ci95([s["frac_below_HB"] for _s, _d, s in corr_seed_data]),
            final_acc_mean=float(np.mean(accs)),
            final_acc_ci=ci95(accs),
        )
        print(f"[r39] corrected arm late-window λref = "
              f"{corr_agg['lambda_late_mean']:.2f} ± {corr_agg['lambda_late_ci']:.2f} "
              f"(n={corr_agg['n_seeds']})   final_acc = "
              f"{corr_agg['final_acc_mean']:.4f} ± {corr_agg['final_acc_ci']:.4f}")

    # ---- aggregate across seeds --------------------------------------------
    # R30 CSV zero-change guard (orchestrator-authorized minimal IO guard):
    # the emitted summary CSV is treated as frozen data — if it already
    # exists, read it back and use it downstream (panel (c), verdict) WITHOUT
    # rewriting the file; compute + write only when it is missing.
    summ_csv = FIG / "cifar100_control_summary.csv"
    if summ_csv.exists():
        summ = pd.read_csv(summ_csv)
        print(f"[guard] {summ_csv} exists — reused verbatim, rewrite skipped")
    else:
        agg_rows = []
        for opt, name, _, _ in present:
            stats = [s for _, _, s in seed_data[opt]]
            n = len(stats)

            lam_means = np.array([s["lambda_late_mean"] for s in stats])
            lam_mins  = np.array([s["lambda_late_min"]  for s in stats])
            frac_hb   = np.array([s["frac_below_HB"]    for s in stats])
            accs      = np.array([s["final_acc"]        for s in stats])

            rec = dict(
                opt=opt,
                n_seeds=n,
                lambda_late_mean=float(np.mean(lam_means)),
                lambda_late_ci=ci95(lam_means),
                lambda_late_min_mean=float(np.mean(lam_mins)),
                frac_below_HB_mean=float(np.mean(frac_hb)),
                frac_below_HB_ci=ci95(frac_hb),
                final_acc_mean=float(np.mean(accs)),
                final_acc_ci=ci95(accs),
            )
            agg_rows.append(rec)

        summ = pd.DataFrame(agg_rows)[
            ["opt", "n_seeds", "lambda_late_mean", "lambda_late_ci",
             "lambda_late_min_mean", "frac_below_HB_mean", "frac_below_HB_ci",
             "final_acc_mean", "final_acc_ci"]
        ]
        summ.to_csv(summ_csv, index=False)
    # single source for panel (c) + verdict, identical in both branches
    agg = {row["opt"]: dict(row) for _i, row in summ.iterrows()}

    print("\n=== CIFAR-100 / ResNet-18 control summary (late window: step >= "
          f"{LATE_STEP}; mean ± 95% CI across seeds) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[csv] {summ_csv}")

    # ---- figure: (a) full-width top row, (b)(c) side-by-side bottom row -----
    fig = plt.figure(figsize=(6.7, 5.2))
    gs_rows = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    # spacer column between (b) and (c): a slim gutter so the (c) ylabel
    # ("late-window λ_max...") cannot read as attached to (b) "CIFAR-100
    # accuracy". 0.24 → 0.12 → 0.06 across revisions (twice halved at user
    # request); the freed width goes to (b).
    gs_bottom = gs_rows[1].subgridspec(1, 3, width_ratios=[1.53, 0.06, 1.0])
    axes = [fig.add_subplot(gs_rows[0]),        # (a) trajectories — full width
            fig.add_subplot(gs_bottom[0, 0]),   # (b) accuracy (wider)
            fig.add_subplot(gs_bottom[0, 2])]   # (c) late-window bars (col 1 = gutter)

    # panel 0: lambda_max trajectories (faint per-seed + bold per-opt mean)
    # TRAJECTORIES: dense _detarpack_d20 (251 pts/run; see module docstring);
    # R39: the corrected arm falls back to its canonical every-100 grid.
    # R30 hierarchy: per-seed curves are TEXTURE (ST seed lw + alpha, same
    # hue as their method), the 4 bold means are the subject.
    ax = axes[0]
    # two-pass draw (same fix as fig 28): ALL faint per-seed lines first, then
    # ALL bold means. SGD vs rank-1 nearly coincide (the paper's punchline), so
    # stacking matters: means get descending zorder by list index -> the
    # baseline (SGD, neutral) sits on top where curves overlap and the later
    # opts stay visible wherever they deviate.
    def _traj(opt):
        return corr_seed_traj if opt == CORR_OPT else seed_traj[opt]

    for opt, _name, c, _m in arms:
        for _seed, df in _traj(opt):
            ax.plot(df.step, df.lambda_max, "-", lw=ST["seed"], color=c,
                    alpha=ST["alpha_seed"], zorder=2)
    # (user round: the rank-1 mean is fully buried under SGD wherever the
    # curves coincide — solid-on-solid with SGD at the higher zorder reads
    # as "only teal + gray exist". Fix: non-solid strokes drawn ABOVE the
    # solid SGD mean, so overlap shows textured strokes riding the slate
    # line and every mean stays identifiable. R39 (fixplan §5.11/§6): four
    # distinguishable styles — SGD '-'/o, legacy rank-1 ':'/x, corrected
    # ARPACK-sensed rank-1 '-.'/s, SAM '--'/^; markers ride the bold means
    # only (markevery), redundant with hue for grayscale print. Applied to
    # BOTH trajectory panels so the single panel-(a) legend stays truthful
    # for panel (b) as well.)
    mean_style = {"sgd": ("-", "o", 5), "rank1": (":", "x", 6),
                  CORR_OPT: ("-.", "s", 4.5), "sam": ("--", "^", 4)}
    for i, (opt, name, c, _m) in enumerate(arms):
        # mean across seeds (aligned on common step grid) — thinned 2.4→1.4
        cols = [df.set_index("step")["lambda_max"].rename(i)
                for i, (_seed, df) in enumerate(_traj(opt))]
        mean_traj = pd.concat(cols, axis=1).mean(axis=1)  # skipna -> robust to misalignment
        ls, mk, z = mean_style[opt]
        # display smoothing: sigma in SAMPLES — 12 on the dense every-20
        # grid; 2.4 on the corrected arm's every-100 grid keeps the same
        # ~240-step physical width (see smooth32 docstring)
        sig = 12.0 if (opt != CORR_OPT or corr_traj_dense) else 2.4
        mean_traj = smooth32(mean_traj, sigma=sig)
        ax.plot(mean_traj.index, mean_traj.values, ls, lw=ST["primary"],
                color=c, label=name, zorder=z, marker=mk,
                markevery=max(1, len(mean_traj) // 10),
                markersize=ST["marker"], markerfacecolor=c,
                markeredgecolor=c)
    # two theory boundaries: very light horizontal rules (theory_light, thin)
    # — reference, not subject; identified by small right-end tags below
    trans_a = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(GD_THRESH, color=THEORY["light"], ls="--", lw=ST["theory"],
               zorder=1.5)
    ax.axhline(HB_THRESH, color=THEORY["light"], ls="-.", lw=ST["theory"],
               zorder=1.5)
    # GD tag at the LEFT end, just below its rule: the right end (old
    # placement) is crossed by the SGD/rank-1 means + seed texture once they
    # settle toward the 20-38 band (x≈4500-5000); the below-20 band at the
    # left end (steps ≲2900) is ink-free on the rendered layer. The HB tag
    # stays right-end (that band is verified empty: curves live at 28-35).
    ax.text(0.015, GD_THRESH * 0.965, fr"$2/\eta = {GD_THRESH:g}$ (GD)",
            transform=trans_a, ha="left", va="top",
            fontsize=FS_TAG, color=THEORY["dark"])
    ax.text(0.985, HB_THRESH * 1.025,
            fr"$2(1+\mu)/\eta = {HB_THRESH:g}$ (HB reference)",
            transform=trans_a, ha="right", va="bottom",
            fontsize=FS_TAG, color=THEORY["dark"])
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_{\mathrm{ref}}$")
    ax.set_yscale("log")
    # revision fix: cap the log axis so the decisive 20–38 threshold band is
    # readable. Uncapped, the y-range runs to ~1.2e4 (init transient) and the
    # band collapses to ~11% of panel height. With ylim (14, 150) every
    # non-transient data point displays well inside the frame (dense-run
    # max outside the step-0/20 spike is 96; only 9 of 3765 points sit
    # above 100), the transient decay enters with headroom below the top
    # spine, and the 2/η=20 → 2(1+μ)/η=38 band keeps ~30% of panel height.
    # The spike itself is clipped — flagged by the annotation below.
    # (Floor 9: SAM's late-window escape descends to λmax ≈ 10.0-10.6 on
    # all 5 dense seeds — the old floor 14 buried 25-39 points/run of the
    # escape band; 9 shows SAM's full below-20 excursion with margin.)
    ax.set_xlim(0, 5000)
    ax.set_ylim(9, 150)
    # ONE frameless method legend for the whole figure, in the empty
    # upper-right band of the top trajectory panel (thresholds are tagged
    # directly on their rules, so they carry no legend entries)
    ax.legend(loc="upper right")
    # clipped-transient annotation (short, top-left; legend occupies top-right)
    ax.annotate(r"init spike $\sim\!10^{4}$ (clipped)",
                xy=(0.04, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=ST["note"], color="0.3",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    # keep the original tick density at the smaller panel size
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    panel_label(ax, "a", "CIFAR-100 / ResNet-18")

    # panel 1: eval accuracy (faint per-seed + bold per-opt mean)
    # TRAJECTORIES: dense _detarpack_d20 (251 pts/run; see module docstring);
    # R39: the corrected arm rides its canonical every-100 metrics (eval_acc
    # present on all its seeds — the panel truthfully shows all four arms).
    # R30 hierarchy: seed curves = faint texture, means = subject; the method
    # legend lives once, in panel (a) — no per-panel legends.
    ax = axes[1]
    # same two-pass draw: faint per-seed for all arms, then bold means with
    # descending zorder (SGD topmost) — the accuracy curves converge at
    # the end and the neutral mean would otherwise be buried under the others
    for opt, _name, c, _m in arms:
        for _seed, df in _traj(opt):
            ax.plot(df.step, df.eval_acc, "-", lw=ST["seed"], color=c,
                    alpha=ST["alpha_seed"], zorder=2)
    for i, (opt, name, c, _m) in enumerate(arms):
        cols = [df.set_index("step")["eval_acc"].rename(i)
                for i, (_seed, df) in enumerate(_traj(opt))]
        mean_traj = pd.concat(cols, axis=1).mean(axis=1)
        ls, mk, z = mean_style[opt]
        sig = 12.0 if (opt != CORR_OPT or corr_traj_dense) else 2.4
        mean_traj = smooth32(mean_traj, sigma=sig)
        ax.plot(mean_traj.index, mean_traj.values, ls, lw=ST["primary"],
                color=c, zorder=z, marker=mk,
                markevery=max(1, len(mean_traj) // 10),
                markersize=ST["marker"], markerfacecolor=c,
                markeredgecolor=c)
    ax.set_xlabel("step")
    ax.set_ylabel("eval accuracy (top-1)")
    ax.set_xlim(0, 5000)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    panel_label(ax, "b", "CIFAR-100 accuracy")

    # panel 2: late-window lambda_max dot-and-whisker ± CI with BOTH
    # threshold rules (R30: bars -> point-range; theory rules very light)
    # R39: FOUR x positions — the corrected ARPACK-sensed arm joins the
    # frozen three-arm summary with its own seed-level t-CI (never pooled
    # steps; same per-seed late-window recipe as the other arms).
    ax = axes[2]
    labels = [name for _opt, name, _c, _m in arms]
    means, cis, colors, markers = [], [], [], []
    for opt, _name, c, m in arms:
        src = corr_agg if opt == CORR_OPT else agg[opt]
        means.append(src["lambda_late_mean"])
        cis.append(src["lambda_late_ci"])
        colors.append(c)
        # per-arm markers, consistent with panel (a): o / x / s / ^
        markers.append(mean_style[opt][1])
    x = np.arange(len(labels))
    # dot-and-whisker: position + uncertainty only, no bar mass
    for xi, (m, c, col, mk) in enumerate(zip(means, cis, colors, markers)):
        ax.errorbar(xi, m, yerr=c, fmt=mk, color=col, ls="none",
                    markersize=ST["marker"], markerfacecolor=col,
                    markeredgecolor=col, elinewidth=ST["errbar"],
                    capsize=ST["capsize"], zorder=3)
    # mean ± CI annotation ABOVE each CI whisker, single-line "m ± c" —
    # same convention as the other summary figures. Two decimals kept: the
    # paper text cites these exact values (29.21±1.90 etc.).
    # (user round: SGD and rank-1 land at near-equal late λ_ref (28.98 vs
    # 28.83), so their centered labels collided in the middle. The points
    # are one slot apart — horizontal nudging cannot separate ~57pt-wide
    # labels — so stagger VERTICALLY instead: rank-1's label rises a full
    # text-height above the +5pt baseline level, clearing SGD's by ~8pt;
    # SGD and SAM keep the +5pt convention. rank-1's lifted label tops out
    # at ≈35.7 data units, still ~2 units below the 38 dash-dot rule.
    # R39: the corrected arm (29.94±0.60) rides the +5pt convention; with four
    # slots the labels moved closer, so rank-1's lift went 15 -> 18pt to clear
    # BOTH neighbors' boxes on the rendered text layer (bbox-verified).
    label_dy = (5, 18, 5, 5)
    for xi, (m, c) in enumerate(zip(means, cis)):
        ax.annotate(f"{m:.2f} ± {c:.2f}",
                    xy=(xi, m + c), xytext=(0, label_dy[xi]),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=ST["note"])
    # theory boundaries: very light rules + right-end tags (as in panel (a))
    trans_c = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(GD_THRESH, color=THEORY["light"], ls="--", lw=ST["theory"],
               zorder=1.5)
    ax.axhline(HB_THRESH, color=THEORY["light"], ls="-.", lw=ST["theory"],
               zorder=1.5)
    ax.text(0.985, GD_THRESH - 0.55, fr"$2/\eta = {GD_THRESH:g}$",
            transform=trans_c, ha="right", va="top",
            fontsize=FS_TAG, color=THEORY["dark"])
    ax.text(0.985, HB_THRESH + 0.55, fr"$2(1+\mu)/\eta = {HB_THRESH:g}$",
            transform=trans_c, ha="right", va="bottom",
            fontsize=FS_TAG, color=THEORY["dark"])
    # headroom above the annotations (~34) and the 38 rule + tag (~41)
    ax.set_ylim(15, 47)
    ax.set_xlim(-0.55, x[-1] + 0.55)
    ax.set_xticks(x)
    # single-line short labels — full names live in the panel-(a) legend
    # (same color coding). R39: four slots (fixplan §6: two-line short
    # labels for the two rank-1 arms).
    short_labels = ["SGD", "rank-1\nlegacy", "rank-1\nARPACK", "SAM"]
    ax.set_xticklabels(short_labels, rotation=0, ha="center")
    # single-line ylabel: the old two-line "(mean ± 95% CI)" second line,
    # at the R30 font scale, overlaps the y tick labels in this narrow
    # panel (verified on the rendered text layer); the CI convention is
    # stated by the per-point "m ± c" annotations and the caption
    ax.set_ylabel(r"late-window $\lambda_{\mathrm{ref}}$", labelpad=3)
    # very faint horizontal grid only (summary panel; trajectories stay clean)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.4, lw=0.35 * 1.29, color=THEORY["light"])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    panel_label(ax, "c", "Late-window separation")

    # tight layout with reduced padding: default (~1.08) leaves large gaps
    # between (a) and the (b)(c) row; pad 0.3 tightens inter-panel whitespace
    # while still clearing titles/tick labels.
    plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
    # R40 atomic publish (tmp + rename): identical savefig code path (same
    # rcParams/pad, byte-faithful rendering), but a paper build can never
    # pick up a half-written PDF from figures/.
    tmp = savefig(fig, "30_cifar100_r18_control.tmp_atomic")
    out = tmp.with_name("30_cifar100_r18_control.pdf")
    os.replace(tmp, out)
    print(f"[fig] {out}")
    report(out, "30_cifar100_r18_control")

    # ---- R39: corrected-arm summary CSV (separate from the frozen 3-arm
    # cifar100_control_summary.csv). Per-seed late λref (step >= 2000 mean
    # of the ARPACK outcome lambda_max), final eval accuracy, and the plant
    # fire rates from plant_per_step.jsonl (n_fired > 0 per step; the
    # substantive window 1-4000 plus the full 5000). Summary row carries
    # the seed-level t-based 95% CI — never pooled steps.
    if have_corr:
        corr_rows = []
        for seed, _df, st in corr_seed_data:
            fr14 = frfull = float("nan")
            pl = RUNS / f"cifar100_r18_{CORR_OPT}_s{seed}" / "plant_per_step.jsonl"
            if pl.exists():
                f14, ff = [], []
                with open(pl) as fh:
                    for line in fh:
                        r = json.loads(line)
                        fired = r.get("n_fired", 0) > 0
                        ff.append(fired)
                        if 1 <= r.get("step", 0) <= 4000:
                            f14.append(fired)
                fr14 = float(np.mean(f14)) if f14 else float("nan")
                frfull = float(np.mean(ff)) if ff else float("nan")
            corr_rows.append(dict(
                seed=seed, late_lambda_ref=st["lambda_late_mean"],
                n_late_steps=st["n_late"], final_eval_acc=st["final_acc"],
                fire_rate_steps_1_4000=fr14, fire_rate_full=frfull))
        lam = np.array([r["late_lambda_ref"] for r in corr_rows])
        acc = np.array([r["final_eval_acc"] for r in corr_rows])
        corr_rows.append(dict(
            seed=f"summary", late_lambda_ref=float(np.mean(lam)),
            lambda_ref_ci95=ci95(lam), n_seeds=len(lam),
            final_eval_acc=float(np.mean(acc)), final_acc_ci95=ci95(acc),
            fire_rate_steps_1_4000=float(np.mean(
                [r["fire_rate_steps_1_4000"] for r in corr_rows])),
            fire_rate_full=float(np.mean(
                [r["fire_rate_full"] for r in corr_rows]))))
        corr_csv = FIG / "cifar100_control_rankkx_summary.csv"
        cols = ["seed", "late_lambda_ref", "lambda_ref_ci95", "n_seeds",
                "n_late_steps", "final_eval_acc", "final_acc_ci95",
                "fire_rate_steps_1_4000", "fire_rate_full"]
        pd.DataFrame(corr_rows).reindex(columns=cols).to_csv(corr_csv, index=False)
        print(f"[r39][csv] {corr_csv}  (corrected arm "
              f"{corr_agg['lambda_late_mean']:.2f} ± {corr_agg['lambda_late_ci']:.2f}, "
              f"n={corr_agg['n_seeds']}; fire-rate window 1-4000: "
              f"{corr_rows[-1]['fire_rate_steps_1_4000']:.3f})")

    # ---- verdict (judged against HB boundary, NOT 2/eta) --------------------
    def rel_to_hb(mean, ci):
        if (not np.isnan(ci)) and mean + ci < HB_THRESH - 1e-9:
            return "below HB"
        if (not np.isnan(ci)) and mean - ci > HB_THRESH + 1e-9:
            return "above HB"
        return "at HB band"

    print("\n=== VERDICT (judged against the heavy-ball boundary "
          f"2(1+mu)/eta = {HB_THRESH:g}, NOT 2/eta = {GD_THRESH:g}) ===")
    for opt, name, _c, _m in present:
        r = agg[opt]
        cls = rel_to_hb(r["lambda_late_mean"], r["lambda_late_ci"])
        print(f"  {name:20s}: lambda_late = {fmt_ci(r['lambda_late_mean'], r['lambda_late_ci']):>14s}  "
              f"[{cls}],  frac<HB = {r['frac_below_HB_mean']:.0%},  "
              f"final_acc = {fmt_ci(r['final_acc_mean'], r['final_acc_ci'])}")
    if have_corr:
        cls = rel_to_hb(corr_agg["lambda_late_mean"], corr_agg["lambda_late_ci"])
        print(f"  {CORR_NAME:20s}: lambda_late = "
              f"{fmt_ci(corr_agg['lambda_late_mean'], corr_agg['lambda_late_ci']):>14s}  "
              f"[{cls}],  frac<HB = {corr_agg['frac_below_HB_mean']:.0%},  "
              f"final_acc = {fmt_ci(corr_agg['final_acc_mean'], corr_agg['final_acc_ci'])}")

    # strong overall verdict only if SAM's CI sits clearly below the others
    sam_rec = agg.get("sam")
    other_lowers = []
    if sam_rec is not None:
        sam_up = sam_rec["lambda_late_mean"] + sam_rec["lambda_late_ci"]
        for opt, *_ in present:
            if opt == "sam":
                continue
            r = agg[opt]
            other_lowers.append(r["lambda_late_mean"] - r["lambda_late_ci"])
        sam_below_others = (len(other_lowers) > 0) and (sam_up < min(other_lowers) - 1e-9)
    else:
        sam_below_others = False

    if sam_below_others:
        print("\n  >> Qualitative ordering SAM < {SGD ≈ rank-1} reproduces at 11.2M scale; "
              "rank-1 does not measurably reduce the fixed-batch reference sharpness "
              "relative to SGD, whereas SAM does.")
    else:
        # honest fallback
        order = sorted(present, key=lambda t: agg[t[0]]["lambda_late_mean"])
        order_str = " < ".join(name for _opt, name, _c, _m in order)
        print(f"\n  >> Observed late-window lambda_max ordering: {order_str}. "
              "(SAM CI not clearly below SGD/rank-1 at the current seed count; "
              f"HB boundary = {HB_THRESH:g}.)")

    # ---- optional: rank-k forward-compat (k-invariance on ResNet-18) --------
    rankk_check()


def rankk_check():
    """If any runs/cifar100_r18_rankk{2,5}_s* exist, print a k-invariance table
    using rank1 dirs for k=1. Pending -> print a note."""
    print("\n=== k-invariance check on ResNet-18 (lambda_late mean ± 95% CI) ===")
    # k=1 from the rank1 dirs
    table = {}
    k1 = discover("rank1", suffix="")   # original power-iter protocol (matches k=2,5 sweeps)
    if k1:
        lam_means = []
        for _seed, csv, _mf in k1:
            df = pd.read_csv(csv).sort_values("step")
            lam = df[df.step >= LATE_STEP].lambda_max.to_numpy(dtype=float)
            if lam.size:
                lam_means.append(float(np.mean(lam)))
        if lam_means:
            table[1] = (np.mean(lam_means), ci95(lam_means), len(lam_means))

    found_any = bool(k1)
    for k in (2, 5):
        recs = []
        for csv in sorted(glob.glob(str(RUNS / f"cifar100_r18_rankk{k}_s*" / "metrics.csv"))):
            d = Path(csv).parent
            if "smoke" in d.name:
                continue
            recs.append(csv)
        if recs:
            found_any = True
            lam_means = []
            for csv in recs:
                df = pd.read_csv(csv).sort_values("step")
                lam = df[df.step >= LATE_STEP].lambda_max.to_numpy(dtype=float)
                if lam.size:
                    lam_means.append(float(np.mean(lam)))
            if lam_means:
                table[k] = (np.mean(lam_means), ci95(lam_means), len(lam_means))

    if not found_any or not table:
        print("  [pending] runs/cifar100_r18_rankk{2,5}_s* not found yet — "
              "k-invariance check on a real CNN deferred until those runs land.")
        return

    print(f"  {'k':>3s}  {'n_seeds':>7s}  {'lambda_late':>16s}")
    for k in sorted(table):
        m, ci, n = table[k]
        print(f"  {k:>3d}  {n:>7d}  {fmt_ci(m, ci):>16s}")
    if {2, 5} - set(table):
        print("  [partial] missing k in " + str(sorted({2, 5} - set(table))) +
              " — rerun after those seeds land.")


if __name__ == "__main__":
    main()
