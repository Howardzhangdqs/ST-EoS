"""Fig 5 (fig:xsam) — X-SAM ablation on CIFAR-10/ResNet-18 (n=3, ARPACK true λ_max).

Replaces the old MLP-based ``28_xsam_ablation.pdf`` (which displayed the
self-gating-clamp variant that we since showed was an artifact, not faithful
X-SAM). The new figure has two panels:

  (a) λ_max trajectories: sgd / sam / xsam_faithful / xsam_clamped, faint
      per-seed + bold mean across n=3 seeds, with the 2/η stability boundary
      as a dashed horizontal reference. log y-axis (the init transient at
      λ_max~600 would otherwise compress the visually-important late regime).
      **Headline visual:** SAM, faithful, and clamped all descend below SGD;
      faithful ≈ clamped ≈ SAM.

  (b) Paired Δλ_max(faithful − sam) bars at mid (1–3k) and late (3–5k) with
      95% Student-t CI error bars and a 0 reference line. **Headline visual:**
      both CIs cross zero → the rank-1 correction does not significantly reduce
      λ_max below SAM in this setting.

Reads:  runs/cifar10_r18_{sgd,sam,xsam_faithful,xsam_clamped}_s{0,1,2}/metrics.csv
Writes: figures/28_xsam_ablation.pdf

Reproducibility: deterministic runs, ARPACK k=3 tol=1e-6 on B=512 (matches the
training-time outcome measurement). Student-t df=2 critical value 4.303 (honest
wide CI for n=3).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    METHOD_COLORS, METHOD_LINESTYLES, THEORY, ST, setup, savefig,
)

setup(scale=1.21 * 8.5 / 8.57)   # R30: source canvas ~1.21x final size (report()-measured); ×8.5/8.57 R38 tag-size unification

RUNS = Path("runs")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

# opt -> (display name, color, linestyle); colors from the global
# METHOD_COLORS registry, family variants separated by the registry's
# redundant linestyle channel (R30: no new hues for variants)
OPTS = [
    ("sgd",            "SGD",                            METHOD_COLORS["sgd"],            METHOD_LINESTYLES["sgd"]),
    ("sam",            "SAM",                            METHOD_COLORS["sam"],            METHOD_LINESTYLES["sam"]),
    # R39 (fixplan §5.15/§7.2): "faithful" retired from display strings —
    # Section E states this is a controlled projector comparison, not a
    # protocol replication. Run-dir key xsam_faithful and the JSON schema
    # key delta_faithful_minus_sam stay unchanged (data identity).
    ("xsam_faithful",  "X-SAM projector (matched-step)", METHOD_COLORS["xsam_faithful"],  METHOD_LINESTYLES["xsam_faithful"]),
    ("xsam_clamped",   "X-SAM projector (clamped)",      METHOD_COLORS["xsam_clamped"],   METHOD_LINESTYLES["xsam_clamped"]),
]

# Auto-discover available seeds — handles n=3 → n=5 → n=10 transitions.
ALL_SEEDS = list(range(10))  # [0..9]


def discover_seeds():
    """Seeds for which ALL 4 opts have ARPACK data."""
    present = []
    for s in ALL_SEEDS:
        if all((RUNS / f"cifar10_r18_{opt}_s{s}/metrics.csv").exists() for opt, _, _, _ls in OPTS):
            ok = True
            for opt, _, _, _ls in OPTS:
                df = pd.read_csv(RUNS / f"cifar10_r18_{opt}_s{s}/metrics.csv")
                if "lambda_max_arpack_ritz" not in df.columns:
                    ok = False
                    break
            if ok:
                present.append(s)
    return present


SEEDS = discover_seeds()

# CIFAR-10/ResNet-18 protocol: lr=0.1, mom=0.9 → 2/eta=20 (the GD stability
# boundary; matches the preregistration-style λ_targ for the plant experiments
# and the EoS reference in the paper).
ETA = 0.1
LAM_TARG = 2.0 / ETA  # = 20.0
HB_TARG = 2.0 * (1.0 + 0.9) / ETA  # = 38.0, heavy-ball 2(1+μ)/η (runs use μ=0.9)

# Phase windows (in step count). Match the analyze_xsam_multiseed.py slicing.
PHASES = [("mid", 1000, 3000), ("late", 3000, 5000)]


def ci95_half(x):
    """Two-sided 95% CI half-width via Student-t (df=n-1).
    Correct for any n>=2 via scipy (e.g. t_{0.975,9}=2.262 for n=10;
    t_{0.975,2}=4.303 for n=3). NB: the previous hardcoded tcrit table only
    covered n<=5 and fell back to the normal-approx Z=1.96 for larger n,
    which made the error bars ~13% too narrow at n=10."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return float("inf") if n == 0 else 0.0
    sd = float(np.std(x, ddof=1))
    # Student-t critical value (matches analyze_xsam_multiseed.py); scipy so it is
    # correct for any n, not just the small-n table previously hardcoded.
    tcrit = float(student_t.ppf(0.975, n - 1))
    return float(tcrit * sd / math.sqrt(n))


def load_traj(opt, seed):
    """Load one seed's metrics.csv, sorted by step. Returns df."""
    csv = RUNS / f"cifar10_r18_{opt}_s{seed}/metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv}")
    return pd.read_csv(csv).sort_values("step").reset_index(drop=True)


def phase_mean_per_seed(opt, lo, hi, col="lambda_max"):
    """Per-seed phase mean of a column. Returns a length-|SEEDS| list."""
    out = []
    for s in SEEDS:
        df = load_traj(opt, s)
        mask = (df["step"] > 0) & (df["step"] >= lo) & (df["step"] < hi)
        vals = df.loc[mask, col].dropna().to_numpy(dtype=float)
        out.append(float(vals.mean()) if vals.size else float("nan"))
    return out


def main():
    # ---- load all trajectories once ----
    seed_data = {}  # opt -> list of (seed, df)
    for opt, _, _, _ls in OPTS:
        rows = []
        for s in SEEDS:
            rows.append((s, load_traj(opt, s)))
        seed_data[opt] = rows

    # ---- paired Δ(faithful − sam) per phase ----
    delta_stats = []
    for pname, lo, hi in PHASES:
        f_per = phase_mean_per_seed("xsam_faithful", lo, hi)
        s_per = phase_mean_per_seed("sam", lo, hi)
        diffs = [ff - ss for ff, ss in zip(f_per, s_per)]
        m = float(np.mean(diffs))
        c = ci95_half(diffs)
        delta_stats.append({
            "phase": pname, "mean": m, "ci": c,
            "per_seed": diffs, "crosses_zero": (m - c <= 0 <= m + c),
        })
        print(f"  Δ(projector − sam) {pname:5s}: "
              f"per-seed {[f'{x:+.2f}' for x in diffs]}  "
              f"mean={m:+.2f} ± {c:.2f}  crosses_0={m - c <= 0 <= m + c}")

    n = len(SEEDS)
    # ---- figure ----
    fig, (ax_lam, ax_delta) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                                            gridspec_kw={"width_ratios": [1.5, 1.0]})

    # ----- (a) λ_max trajectories: bold seed-mean + 95% CI band -----
    # spaghetti per-seed lines removed (40 lines dominated the panel); the CI
    # band carries the seed variability instead. A 5-pt (≈100-step) centered
    # rolling mean tames per-step noise without distorting the plateaus.
    # R30: CI band = ST["alpha_ci"] texture, mean = ST["primary"], family
    # variants distinguished by the registry linestyle; reference lines in
    # neutral theory gray. No grid on the trajectory panel.
    SMOOTH = 5
    tcrit = float(student_t.ppf(0.975, n - 1))
    for opt, name, c, ls in OPTS:
        cols = [df.set_index("step")["lambda_max"].rename(i)
                for i, (_seed, df) in enumerate(seed_data[opt])]
        wide = pd.concat(cols, axis=1)
        steps = wide.index.to_numpy()
        mean_arr = np.asarray(wide.mean(axis=1).rolling(
            SMOOTH, center=True, min_periods=1).mean())
        half = np.asarray((tcrit * wide.std(axis=1, ddof=1) / math.sqrt(n)
                           ).rolling(SMOOTH, center=True, min_periods=1).mean())
        lo = np.maximum(mean_arr - half, 1e-1)  # guard for log axis
        ax_lam.fill_between(steps, lo, mean_arr + half,
                            color=c, alpha=ST["alpha_ci"], lw=0, zorder=2)
        ax_lam.plot(steps, mean_arr, ls, lw=ST["primary"],
                    color=c, label=name, zorder=3)
    # 2/η (GD) reference + the OPERATIVE heavy-ball boundary 2(1+μ)/η=38 —
    # these runs use momentum μ=0.9, so the stability boundary the plateaus
    # should be judged against is the heavy-ball one; both drawn, same
    # convention as fig 30 panel (a) (GD dashed, heavy-ball dash-dot bold).
    ax_lam.axhline(LAM_TARG, color=THEORY["dark"], ls="--", lw=ST["theory"],
                   label=fr"$2/\eta = {LAM_TARG:g}$")
    ax_lam.axhline(HB_TARG, color=THEORY["dark"], ls="-.", lw=ST["theory"],
                   label=fr"$2(1+\mu)/\eta = {HB_TARG:g}$ (heavy-ball reference)")
    ax_lam.set_xlabel("step")
    ax_lam.set_ylabel(r"$\lambda_{\mathrm{ref}}$ (ARPACK, true)")
    ax_lam.set_yscale("log")
    # clip the log axis to 8–10^3: focuses the view on the SAM/SGD plateau
    # band (~10–80), cutting the empty lower decades (per user request)
    ax_lam.set_ylim(8.0, 1000.0)
    ax_lam.set_xlim(0, 5000)
    ax_lam.set_title(rf"\textbf{{(a)}}  CIFAR-10 / ResNet-18  $n={n}$ seeds", color="#1A1D21")
    # fontsize: rc legend (7.0 final) — the 6-entry legend keeps its
    # upper-right position; the entries have no natural merge (4 methods +
    # 2 distinct reference lines), so it stays 6 rows (R30 decision).
    # R45b (user): legend raised to just above the axes top (y=1.04), right
    # edge anchored to the axes right (x=1.0), out of the plot area.
    ax_lam.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.04))
    ax_lam.set_axisbelow(True)
    ax_lam.grid(False)   # trajectory panel: no grid (R30)

    # ----- (b) paired Δ(faithful − sam) as a forest plot -----
    # point estimate + horizontal CI whiskers per phase; lighter than bars and
    # visually consistent with the line-based panel (a). Faithful X-SAM green.
    labels = [d["phase"] for d in delta_stats]
    means = [d["mean"] for d in delta_stats]
    cis = [d["ci"] for d in delta_stats]
    x = np.arange(len(labels))
    FOREST_C = METHOD_COLORS["xsam_faithful"]
    for xi, m, c in zip(x, means, cis):
        ax_delta.plot([m - c, m + c], [xi, xi], "-", lw=ST["primary"],
                      color=FOREST_C, solid_capstyle="butt", zorder=2)
        for xw in (m - c, m + c):  # end caps
            ax_delta.plot([xw, xw], [xi - 0.07, xi + 0.07], "-",
                          lw=ST["primary"], color=FOREST_C, zorder=2)
        ax_delta.plot([m], [xi], "o", ms=6, color=FOREST_C,
                      mec=THEORY["dark"], mew=0.7, zorder=3)
        # value text at the lower-RIGHT of the dot — a centered label crossed
        # the dashed vertical zero line; starting ~7pt right of the dot stays
        # clear of x=0 and inside the right spine (both means ≈ +0.6)
        ax_delta.annotate(fr"${m:+.2f} \pm {c:.2f}$",
                          xy=(m, xi), xytext=(7, -10),
                          textcoords="offset points",
                          va="top", ha="left", fontsize=9)
    ax_delta.axvline(0.0, color=THEORY["dark"], ls="--", lw=ST["theory"], zorder=1)
    ax_delta.set_yticks(x)
    ax_delta.set_yticklabels([f"{lab}  ({lo // 1000:g}–{hi // 1000:g}k)"
                              for lab, _n, lo, hi in
                              zip(labels, *zip(*PHASES))])
    ax_delta.set_xlabel(r"paired $\Delta\lambda_{\mathrm{ref}}$  (projector $-$ SAM)")
    ax_delta.set_title(rf"\textbf{{(b)}}  Paired rank-1 effect ($n={n}$)", color="#1A1D21")
    # headroom so whiskers + annotations stay inside the panel
    y_max = max(m + c for m, c in zip(means, cis))
    y_min = min(m - c for m, c in zip(means, cis))
    pad = 0.28 * (y_max - y_min)
    ax_delta.set_xlim(y_min - pad, y_max + 1.5 * pad)
    ax_delta.set_ylim(-0.6, len(labels) - 0.4)
    ax_delta.set_axisbelow(True)   # grid behind data ink (forest/whisker panel)
    ax_delta.grid(axis="x", alpha=0.35, lw=0.4, color=THEORY["light"])

    plt.tight_layout()
    out = savefig(fig, "28_xsam_ablation")
    print(f"\n[fig] {out}")

    # ---- also dump the underlying numbers for the caption / paper text ----
    summary = {
        "lambda_max_phase_means_per_opt": {
            opt: {p[0]: {"per_seed": phase_mean_per_seed(opt, p[1], p[2]),
                          "mean": float(np.mean(phase_mean_per_seed(opt, p[1], p[2]))),
                          "ci": ci95_half(phase_mean_per_seed(opt, p[1], p[2]))}
                  for p in [("early", 0, 1000), ("mid", 1000, 3000), ("late", 3000, 5000)]}
            for opt, _, _, _ls in OPTS
        },
        "delta_faithful_minus_sam": delta_stats,
    }
    import json
    json_path = FIG / "28_xsam_ablation_summary.json"
    json_path.write_text(__import__("json").dumps(summary, indent=2, default=str))
    print(f"[json] {json_path}")


if __name__ == "__main__":
    main()
