"""Full-rank linearized-SAM control — review round 3 (rank vs requery).

linSAM applies g ← g + (ρ/‖g‖)·H_B g — the FIRST-ORDER expansion of SAM with
the same ρ = 0.05, but WITHOUT the perturbed-point re-query.  The adjudication:

  * late λmax(linSAM) ≈ λmax(SGD)   → the first-order all-direction
    correction alone does NOT carry SAM's effect → the operative mechanism
    is perturb-and-requery, not rank;
  * late λmax(linSAM) ≈ λmax(SAM)   → rank/globality carries the effect
    (the eigenspace-edit family merely lacks rank);
  * in between                      → both contribute (quantified by the
    paired Δs).

Reads cifar100_r18_linsam_s*, with cifar100_r18_sgd_detarpack_s* and
cifar100_r18_sam_detarpack_s* as references.  Late window step ≥ 2000
(matching sub:scale).  Also reports linSAM's per-step batch sharpness
lam_step (train-mode Hessian byproduct) and correction diagnostics.

Outputs: figures/linsam_summary.csv + figures/41_linsam.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from steos.analysis.figure_style import setup, C, THEORY, ST
import numpy as np
import pandas as pd
from scipy import stats

LATE_STEP = 2000
# round 5: prefer the n=5 mb reruns (same trajectories — verified per-step
# bit-identical to the round-3 linsam runs; they additionally carry the
# batch-sharpness and Taylor columns).  Falls back to legacy prefixes.
def _pfx(cand, legacy):
    return cand if any(Path("runs").glob(cand + "*")) else legacy

LIN_PFX = _pfx("cifar100_r18_linsam_mb_s", "cifar100_r18_linsam_s")
SGD_PFX, SAM_PFX = "cifar100_r18_sgd_detarpack_s", \
    "cifar100_r18_sam_detarpack_s"


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_ci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float(d.mean()), 0.0, float(d.mean()), 0.0
    m = float(d.mean())
    h = float(stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n))
    return m, h, m - h, m + h


def load(prefix, col="lambda_max"):
    out = {}
    for d in sorted(Path("runs").glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            out[int(s)] = pd.read_csv(d / "metrics.csv")
    return out


def late_mean(df, col="lambda_max"):
    m = df[df["step"] >= LATE_STEP]
    return float(m[col].mean())


def main():
    setup(scale=1.2386)   # R30: source canvas ~1.24x final size (report()-measured)
    sgd, sam, lin = load(SGD_PFX), load(SAM_PFX), load(LIN_PFX)
    if not lin:
        raise SystemExit("no linsam runs found")
    print(f"sgd seeds {sorted(sgd)} | sam seeds {sorted(sam)} | linsam seeds {sorted(lin)}")

    rows = []
    for name, runs in (("SGD", sgd), ("SAM", sam), ("linSAM", lin)):
        per = [late_mean(df) for df in runs.values()]
        m, h = ci95(per)
        rows.append(dict(arm=name, late_lmax=m, ci95=h, n=len(per)))
        print(f"{name:7s} late λmax {m:6.2f} ± {h:4.2f}  (n={len(per)})")

    common = sorted(set(lin) & set(sgd) & set(sam))
    if common:
        d_sgd = [late_mean(lin[s]) - late_mean(sgd[s]) for s in common]
        d_sam = [late_mean(lin[s]) - late_mean(sam[s]) for s in common]
        for label, d in (("Δ(linSAM−SGD)", d_sgd), ("Δ(linSAM−SAM)", d_sam)):
            m, h, lo, hi = paired_ci(d)
            rows.append(dict(arm=label, late_lmax=m, ci95=h, n=len(d)))
            print(f"{label} paired = {m:+.2f}  90% CI [{lo:+.2f},{hi:+.2f}]  seeds {common}")

    # per-step linSAM diagnostics (train-mode batch sharpness + correction)
    diag = {"lam_step": [], "corr_rel": [], "cos_g_hg": []}
    for s, df in sorted(lin.items()):
        jl = Path("runs") / f"{LIN_PFX}{s}" / "plant_per_step.jsonl"
        if not jl.exists():
            continue
        recs = [json.loads(l) for l in jl.read_text().splitlines()]
        recs = recs[LATE_STEP:]
        diag["lam_step"].append(float(np.mean([r["lam_step"] for r in recs])))
        diag["corr_rel"].append(float(np.mean(
            [r["corr_norm"] / max(r["gnorm"], 1e-30) for r in recs])))
        diag["cos_g_hg"].append(float(np.mean([r["cos_g_hg"] for r in recs])))
    for k, v in diag.items():
        if v:
            m, h = ci95(v)
            rows.append(dict(arm=f"linSAM {k} (late mean)", late_lmax=m, ci95=h, n=len(v)))
            print(f"linSAM {k:10s} late mean {m:7.3f} ± {h:.3f}")

    # R30 adjudication 2 (read-if-exists guard): figures/linsam_summary.csv is
    # FROZEN at its on-disk state — rewriting it only reintroduces ~1e-13
    # scipy/pandas fp drift (md5 must stay stable across R30 regens).  The
    # figure below does not read this CSV (trajectories come from runs/), so
    # the guard is a pure skip-rewrite; the file is only minted when missing.
    linsam_csv = Path("figures/linsam_summary.csv")
    if linsam_csv.exists():
        print(f"[guard] {linsam_csv} exists — keeping frozen file, skipping "
              "rewrite (R30 adjudication 2)")
    else:
        pd.DataFrame(rows).to_csv(linsam_csv, index=False)
        print(f"=> {linsam_csv}")

    # ---- figure: trajectories ----
    # R30 style: palette SGD=neutral slate / SAM=teal / linSAM=purple,
    # theory refs in neutral gray, stroke hierarchy from ST (seed band =
    # alpha_ci texture, mean = primary).  Title compressed to a 3-6 word
    # conclusion (protocol moved to the paper caption).
    fig, ax = plt.subplots(figsize=(4.3, 2.45))
    # R39 grayscale-redundancy lane (fixplan_r15 §6): distinct linestyle +
    # marker per arm (SGD '-'/o, SAM '--'/^, linSAM '-.'/s) on the bold
    # mean curves so the three arms stay separable in grayscale print;
    # min–max bands are unchanged (hue + alpha texture only).
    STYLES = {"SGD": ("-", "o"), "SAM": ("--", "^"), "linSAM": ("-.", "s")}
    for name, runs, c in (("SGD", sgd, C["neutral"]), ("SAM", sam, C["sam"]),
                          ("linSAM", lin, C["linsam"])):
        if not runs:
            continue
        dfs = [df.sort_values("step") for df in runs.values()]
        steps = dfs[0]["step"].to_numpy()
        arr = np.full((len(dfs), len(steps)), np.nan)
        for i, df in enumerate(dfs):
            arr[i] = df["lambda_max"].to_numpy()
        ls, mkr = STYLES[name]
        ax.plot(steps, np.nanmean(arr, 0), color=c, ls=ls, marker=mkr,
                ms=ST["marker"], markevery=max(1, len(steps) // 10),
                lw=ST["primary"], label=name)
        ax.fill_between(steps, np.nanmin(arr, 0), np.nanmax(arr, 0),
                        color=c, alpha=ST["alpha_ci"], lw=0)
    ax.axvline(LATE_STEP, color=THEORY["light"], ls=":", lw=ST["theory"])
    ax.axhline(2 * (1 + 0.9) / 0.1, color=THEORY["dark"], ls="--",
               lw=ST["theory"])
    # R38 (user): label moved to the RIGHT end of the line, with the same
    # white-halo treatment as fig 40's boundary labels (curves pass behind
    # the text, the dash line itself is untouched).
    ax.annotate(r"$2(1{+}\mu)/\eta = 38$", (0.98, 39),
                xycoords=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=9.91,
                path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
    ax.set_xlabel("step"); ax.set_ylabel(r"$\lambda_{\mathrm{ref}}$ (fixed eval batch)")
    ax.set_ylim(0, 80)
    # R38 (user): legend nudged up a few px past the axes top edge (title is
    # centered, right corner stays clear). (R45b: restored after the R45
    # mis-target was rolled back — the intended figure was 28_xsam_ablation.)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03)); ax.grid(False)
    ax.set_title(r"linSAM drifts sharp-ward", fontsize=8.5 * 1.2386)
    fig.tight_layout()
    # R39 atomic regeneration (another session may be compiling the paper):
    # save to <name>.tmp.pdf first, mv over the live PDF only after success.
    outp = Path("figures/41_linsam.pdf")
    tmp = outp.with_suffix(".tmp.pdf")
    fig.savefig(tmp, bbox_inches="tight")
    tmp.replace(outp)
    print(f"=> {outp}")


if __name__ == "__main__":
    main()
