"""Regime-transition bridge: batch-size × momentum controlled sweep — review round 6.

The round-6 review's priority-1 ask: the deterministic (full-batch MLP) and
stochastic (bs=128, mu=0.9, ResNet-18) regimes were demonstrated at two
distant operating points, leaving the claimed mechanism transition
uncharacterized. This script analyzes the controlled sweep that bridges them:
B ∈ {2048, 512, 128, 32} × mu ∈ {0, 0.9} on the SAME network / task /
protocol (5000 steps, deterministic ARPACK reference, multibatch batch
sharpness), arms {sgd-sensing reference, rank-1 damping, rank-1
amplification α=0.5}, n=3 seeds. The (128, 0.9) anchor cell REUSES the
round-3/5 runs (sgd_detarpack_mb2 / rankkx_d1_mb / amp05_mb).

Per cell it extracts the reviewer's phase-transition quantities:
  - gradient-energy share of the edited top-1 subspace  (plant jsonl:
    cos_g_v² in d1; (gdotv/gnorm)² in amp05) — the per-step authority;
  - damping outcome      = late reference λ (d1)  − sgd, paired over seeds;
  - amplification outcome= late reference λ (amp) − sgd, paired;
  - batch sharpness E_B[g_BᵀH_Bg_B/‖g_B‖²] (sgd arm, lhg_bs_mean);
  - reference λ_max (ARPACK fixed-batch, all arms).

Outputs (missing cells/arms are skipped with a warning — partial-safe):
  figures/regime_sweep_summary.csv
  figures/45_regime_sweep.pdf   (a: grad share vs B; b: paired Δλ vs B;
                                  c: batch sharpness vs B + EoSS plateaus;
                                  d: reference λ vs B + boundary guides)

R30 aesthetic redesign (aesthetic_diagnosis_r29.md p12, 4.5 -> 7.0):
sign (damping=blue / amplification=rose) and momentum (µ=0.9 filled /
µ=0 open) are now ORTHOGONAL encodings; SGD stays neutral slate (iron
rule; the pre-R30 tab:blue is the deviation being fixed); theory
boundaries/plateaus unified gray; per-panel legends replaced by ONE
figure-level frameless shared key; top-row duplicate "batch size B"
xlabel removed. Data/CSV values unchanged. CSV writes are guarded
(read-if-exists): an existing summary is loaded as-is and never
rewritten (zero-churn red line).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from steos.analysis.figure_style import setup, C, THEORY, ST, panel_label
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

LATE_STEP = 2000
ETA = 0.1
GD = 2 / ETA                       # 20  — momentum-free boundary
SEEDS = (0, 1, 2)

# (bs, mu, tag) — cell order for plotting; anchor maps to legacy run names.
CELLS = [
    (2048, 0.9, "m09", "legacy-none"),
    (512,  0.9, "m09", "legacy-none"),
    (128,  0.9, "m09", "legacy-anchor"),          # rounds 3+5
    (32,   0.9, "m09", "legacy-none"),
    (2048, 0.0, "m0",  "legacy-none"),
    (512,  0.0, "m0",  "legacy-none"),
    (128,  0.0, "m0",  "legacy-none"),
    (32,   0.0, "m0",  "legacy-none"),
]

# run-name templates per arm. {bs},{mt} for new cells; anchor overrides.
ARM_TPL = {
    "sgd":   "cifar100_r18_bs{bs}_{mt}_sgd_s{s}",
    "d1":    "cifar100_r18_bs{bs}_{mt}_d1_s{s}",
    "amp":   "cifar100_r18_bs{bs}_{mt}_amp05_s{s}",
}
ANCHOR_TPL = {
    "sgd":   "cifar100_r18_sgd_detarpack_mb2_s{s}",
    "d1":    "cifar100_r18_rankkx_d1_mb_s{s}",
    "amp":   "cifar100_r18_amp05_mb_s{s}",
}


def run_name(bs: int, mt: str, arm: str, seed: int, anchor: bool) -> str:
    tpl = ANCHOR_TPL if anchor else ARM_TPL
    return tpl[arm].format(bs=bs, mt=mt, s=seed)


def ci95(xs):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_ci(d, level=0.90):
    """Paired mean difference ± t-CI (ddof=1).  Default 0.90 keeps the
    historical delta_ci/delta_lo/delta_hi CSV columns; the R27 figure pass
    adds a 95% column (delta_ci95) and renders panel (b) at 0.95."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(d) < 2:
        m = float(d.mean())
        return m, 0.0, m
    m = float(d.mean())
    h = float(stats.t.ppf(0.5 + level / 2, len(d) - 1)
              * d.std(ddof=1) / np.sqrt(len(d)))
    return m, h, m - h


WINDOWS = [("late", LATE_STEP, None), ("mid", 200, LATE_STEP)]


def _tint(hex_color: str, f: float = 0.45) -> str:
    """Blend a #RRGBB color toward white by fraction f (the µ=0 channel)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(v + (255 - v) * f) for v in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def late_metrics(run: str, lo=LATE_STEP, hi=None):
    """Window means from metrics.csv: reference λ, batch sharpness,
    fired fraction. (lo, hi) = [lo, hi) step bounds; late window default."""
    f = RUNS / run / "metrics.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df["step"] >= lo]
    if hi is not None:
        df = df[df["step"] < hi]
    out = {"ref_lam": float("nan"), "bsharp": float("nan"),
           "fired_frac": float("nan"), "curmb_lam": float("nan")}
    if "lambda_max" in df and df["lambda_max"].notna().any():
        out["ref_lam"] = float(df["lambda_max"].mean())
    if "lam1_curmb" in df and df["lam1_curmb"].notna().any():
        out["curmb_lam"] = float(df["lam1_curmb"].mean())
    if "lhg_bs_mean" in df and df["lhg_bs_mean"].notna().any():
        out["bsharp"] = float(df["lhg_bs_mean"].mean())
    if "fired_frac" in df and df["fired_frac"].notna().any():
        out["fired_frac"] = float(df["fired_frac"].mean())
    return out


def grad_share(run: str, arm: str, lo=LATE_STEP, hi=None):
    """Window gradient-energy share of the edited top-1 subspace from
    the per-step plant jsonl (d1: cos_g_v; amp: gdotv/gnorm)."""
    f = RUNS / run / "plant_per_step.jsonl"
    if not f.exists():
        return float("nan")
    shares = []
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = r.get("step", 0)
            if st < lo or (hi is not None and st >= hi):
                continue
            if arm == "d1" and "cos_g_v" in r and r["cos_g_v"]:
                shares.append(sum(c * c for c in r["cos_g_v"]))
            elif arm == "amp" and "gdotv" in r and r.get("gnorm"):
                c = r["gdotv"] / r["gnorm"]
                shares.append(c * c)
    return float(np.mean(shares)) if shares else float("nan")


def collect_summary():
    """Full recomputation from runs/ (only invoked when the summary CSV is
    missing — see the read-if-exists guard in main)."""
    rows = []
    for bs, mu, mt, legacy in CELLS:
        anchor = legacy == "legacy-anchor"
        # per-window per-arm per-seed metrics (prereg addendum: the fixed-step
        # design confounds epochs across B, so both the preregistered late
        # window and a mid window 200 <= step < 2000 are emitted; B=2048
        # figure points use the mid window — see R6_PREREG.md addendum).
        for wname, lo, hi in WINDOWS:
            per = {}
            for arm in ("sgd", "d1", "amp"):
                per[arm] = {}
                for s in SEEDS:
                    rn = run_name(bs, mt, arm, s, anchor)
                    m = late_metrics(rn, lo, hi)
                    if m is None:
                        continue
                    m["share"] = (grad_share(rn, arm, lo, hi)
                                  if arm in ("d1", "amp") else float("nan"))
                    per[arm][s] = m
            for arm in ("sgd", "d1", "amp"):
                seeds = sorted(per[arm])
                if not seeds:
                    if wname == "late":
                        print(f"[warn] cell bs={bs} mu={mu} arm={arm}: no runs found")
                    continue
                lam_m, lam_ci = ci95([per[arm][s]["ref_lam"] for s in seeds])
                bs_m, bs_ci = ci95([per[arm][s]["bsharp"] for s in seeds])
                sh_m, sh_ci = ci95([per[arm][s]["share"] for s in seeds])
                ff_m, _ = ci95([per[arm][s]["fired_frac"] for s in seeds])
                row = dict(bs=bs, mom=mu, arm=arm, window=wname, n=len(seeds),
                           ref_lam=lam_m, ref_lam_ci=lam_ci,
                           bsharp=bs_m, bsharp_ci=bs_ci,
                           curmb_lam=float(np.nanmean(
                               [per[arm][s]["curmb_lam"] for s in seeds])),
                           grad_share=sh_m, grad_share_ci=sh_ci,
                           fired_frac=ff_m, anchor=anchor)
                # paired Δ vs sgd on shared seeds
                shared = [s for s in seeds if s in per["sgd"]]
                if arm != "sgd" and shared and per["sgd"]:
                    d = [per[arm][s]["ref_lam"] - per["sgd"][s]["ref_lam"]
                         for s in shared]
                    m, h, lo2 = paired_ci(d)          # 90% (legacy columns)
                    m95, h95, lo95 = paired_ci(d, level=0.95)  # R27
                    row.update(delta_vs_sgd=m, delta_ci=h, delta_lo=lo2,
                               delta_hi=m + h, n_shared=len(shared),
                               delta_ci95=h95, delta_lo95=lo95,
                               delta_hi95=m95 + h95)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="45_regime_sweep")
    args = ap.parse_args()
    setup(scale=1.2489)   # project style; re-measured via report() below

    FIGS.mkdir(exist_ok=True)
    csv = FIGS / "regime_sweep_summary.csv"
    if csv.exists():
        # R30 zero-churn guard: the committed summary IS the data source;
        # read it and never rewrite (analysis results are frozen).
        df = pd.read_csv(csv)
        print(f"[guard] {csv} exists — loaded {len(df)} rows, not rewritten")
    else:
        df = collect_summary()
        df.to_csv(csv, index=False)
        print(df.to_string(index=False))
        print(f"\nwrote {csv}")

    # ---------------- phase figure ----------------
    # (user round: subplot heights lowered a notch — 5.3→4.6in total, panels
    # each ~13% shorter at unchanged width; the two legend rows below live
    # outside the canvas and re-expand via tight bbox, unaffected.)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6))
    mus = [0.9, 0.0]
    # R30 orthogonal encoding, R38 revision (user: same-hue filled/open
    # alone was not readable at this marker size) —
    #   SIGN stays hue + shape (caption-locked):
    #     damping -> damping blue, CIRCLES 'o'   (caption "damping (circles)")
    #     amplification -> rose, TRIANGLES '^'   (caption "amplification
    #                                               (triangles)")
    #     SGD -> neutral slate, circles 'o' (iron rule: SGD is never a hue;
    #            the damping/SGD circles never co-occur: (a)/(b) plot only
    #            the damping/amplification arms, (c)/(d) only SGD)
    #   MOMENTUM now leads with LINE STYLE, backed by two redundant cues:
    #     µ=0.9 -> SOLID line + filled marker + full color
    #     µ=0   -> DASHED line + open marker + light tint of the same hue
    #    (grayscale-safe and CB-safe: dark/light + solid/dashed survive both)
    ARM_STYLE = {"sgd": (C["neutral"], "o"), "d1": (C["damping"], "o"),
                 "amp": (C["amplification"], "^")}
    mu_name = {0.9: r"$\mu{=}0.9$", 0.0: r"$\mu{=}0$"}

    # window selection per prereg addenda: B≤128 keeps the preregistered
    # late window (still training at 5000 steps); B≥512 memorizes by then
    # (51+ epochs), so the mid window 200–2000 is the training regime.
    def sel(mu, arm):
        sub = df[(df.mom == mu) & (df.arm == arm)]
        return pd.concat([sub[(sub.bs <= 128) & (sub.window == "late")],
                          sub[(sub.bs >= 512) & (sub.window == "mid")]]
                         ).sort_values("bs")

    def sweep(ax, mu, arm, y, yerr):
        sub = sel(mu, arm)
        if not len(sub) or y not in sub:
            return
        c, mk = ARM_STYLE[arm]
        solid = (mu == 0.9)
        col = c if solid else _tint(c)
        ax.errorbar(sub.bs, sub[y], yerr=sub[yerr], marker=mk,
                    ls="-" if solid else "--", color=col,
                    lw=ST["secondary"], elinewidth=ST["errbar"],
                    capsize=ST["capsize"], capthick=ST["errbar"],
                    markersize=ST["marker"], markeredgewidth=ST["errbar"],
                    markeredgecolor=col,
                    markerfacecolor=col if solid else "none")

    # (a) gradient share vs B
    ax = axs[0, 0]
    for mu in mus:
        for arm in ("d1", "amp"):
            sweep(ax, mu, arm, "grad_share", "grad_share_ci")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylabel("grad-energy share of\nedited top-1 subspace")
    panel_label(ax, "a", "Per-step actuation")

    # (b) paired Δλ vs B
    ax = axs[0, 1]
    for mu in mus:
        for arm in ("d1", "amp"):
            # R27: panel (b) renders the ordinary paired difference at 95%
            # (delta_ci95); the 90% columns remain in the CSV (TOST layer).
            sweep(ax, mu, arm, "delta_vs_sgd", "delta_ci95")
    ax.axhline(0, color=THEORY["light"], lw=ST["theory"])
    ax.set_xscale("log", base=2)
    ax.set_ylabel(r"paired $\Delta\lambda_{\mathrm{ref}}$ vs SGD" "\n"
                  r"(95\% CI)")
    panel_label(ax, "b", "Sign-control outcome")

    # (c) batch sharpness vs B + EoSS plateaus (theory refs: unified gray)
    ax = axs[1, 0]
    for mu in mus:
        sweep(ax, mu, "sgd", "bsharp", "bsharp_ci")
        # momentum plateaus 2(1∓µ)/η — light-gray dashed, never method hue
        # (R38: kept verbatim per user — µ=0 draws 2/η twice, unchanged)
        for pl in (2 * (1 - mu) / ETA, 2 * (1 + mu) / ETA):
            ax.axhline(pl, color=THEORY["light"], ls="--", lw=ST["theory"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel(r"batch sharpness" "\n"
                  r"$\mathrm{E}_B[g_B^\top H_B g_B / \|g_B\|^2]$")
    panel_label(ax, "c", "EoSS batch sharpness")

    # (d) reference λ vs B + boundaries (2/η dark dotted; 2(1+µ)/η light)
    ax = axs[1, 1]
    for mu in mus:
        sweep(ax, mu, "sgd", "ref_lam", "ref_lam_ci")
    ax.axhline(GD, color=THEORY["dark"], ls=":", lw=ST["theory"])
    ax.axhline(2 * 1.9 / ETA, color=THEORY["light"], ls="--", lw=ST["theory"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel(r"reference $\lambda_{\mathrm{ref}}$" "\n" r"(fixed batch)")
    panel_label(ax, "d", "Reference sharpness")

    # ONE figure-level frameless shared key (per-panel legends deleted).
    # Rows: sign/momentum arms + the two gray theory reference styles.
    def key_line(color, ls, marker=None, filled=True):
        return Line2D(
            [], [], color=color, ls=ls, lw=ST["secondary"],
            marker=marker, markersize=ST["marker"] if marker else None,
            markeredgecolor=color,
            markerfacecolor=color if filled and marker else "none",
        )

    # R38 (user): two stacked legend rows, EACH CENTERED —
    #   row 1: damping | amplif. | SGD                      (the three arms)
    #   row 2: µ=0.9 | µ=0 | plateaus 2(1∓µ)/η | 2/η        (the other four)
    fig.legend([key_line(C["damping"], "-", "o", True),
                key_line(C["amplification"], "-", "^", True),
                key_line(C["neutral"], "-", "o", True)],
               ["damping", "amplif.", "SGD"],
               loc="upper center", bbox_to_anchor=(0.5, -0.008),
               ncol=3, frameon=False, handletextpad=0.5,
               columnspacing=1.2, handlelength=1.8)
    # (user round: row 2 pulled up closer to row 1 — -0.085→-0.06 trims the
    # inter-row ink gap from ~16.6pt to ~7pt at the 4.6in canvas; row 1
    # anchor untouched.)
    fig.legend([key_line(C["neutral"], "-", "o", True),
                key_line(_tint(C["neutral"]), "--", "o", False),
                Line2D([], [], color=THEORY["light"], ls="--", lw=ST["theory"]),
                Line2D([], [], color=THEORY["dark"], ls=":", lw=ST["theory"])],
               [mu_name[0.9], mu_name[0.0],
                r"plateaus $2(1{\mp}\mu)/\eta$", r"$2/\eta$"],
               loc="upper center", bbox_to_anchor=(0.5, -0.06),
               ncol=4, frameon=False, handletextpad=0.5,
               columnspacing=1.2, handlelength=1.8)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.925, bottom=0.08,
                        wspace=0.36, hspace=0.27)  # R38: mean of the 0.18 /
                        # 0.36 passes — tighter than 2/3, no label collision.
                        # bottom: xlabels only — both legend rows now live
                        # OUTSIDE the canvas (tight bbox re-expands for them).
    out = FIGS / f"{args.out_prefix}.pdf"
    # R29 clipping lesson: usetex tight-bbox under-measures rotated y-label
    # ink (pad 0.02 shaved "subspace" descenders). 0.06in clears it; the
    # four-edge render check below is the gate — raise the pad if any edge
    # shows ink within ~2pt of the CropBox.
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
