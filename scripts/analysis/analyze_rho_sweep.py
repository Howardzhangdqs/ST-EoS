"""ρ-sweep: full SAM vs linearized SAM + Taylor-residual scaling (round 5).

Reviewer priority 3: (a) does the SAM-vs-linSAM sharpness separation shrink
as ρ→0 and from which ρ does it appear? (b) how does the NONLINEAR re-query
content  ‖∇L_B(θ+ρĝ) − (g+ρH_Bĝ)‖/‖g‖  (taylor_res_rel, logged by the
training script at every measurement point) scale with ρ — the first-order
part (taylor_d_lin) should scale ~ρ, a second-order-dominated residual ~ρ².

Arms: cifar100_r18_{sam,linsam}_rho{R}_s{0,1,2} for R in {0.005,0.01,0.02,0.1}
plus ρ=0.05 from the round-5 mb reruns (sam_mb / linsam_mb, cadence 200) and
the plain SGD baseline (sgd_detarpack).  Late window step ≥ 2000.

Outputs: figures/rho_sweep_summary.csv, figures/44_rho_sweep.pdf
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from steos.analysis.figure_style import setup, C, THEORY, ST
import numpy as np
import pandas as pd
from scipy import stats

LATE_STEP = 2000
ETA = 0.1
GD = 2 / ETA

RHO_NEW = [0.005, 0.01, 0.02, 0.1]
# (arm, {ρ: prefix}) — ρ=0.05 comes from the round-5 mb reruns (taylor logged)
ARM_RHOS = {
    "sam": {**{r: f"cifar100_r18_sam_rho{r}_s" for r in RHO_NEW},
            0.05: "cifar100_r18_sam_mb_s"},
    "linsam": {**{r: f"cifar100_r18_linsam_rho{r}_s" for r in RHO_NEW},
               0.05: "cifar100_r18_linsam_mb_s"},
}
SGD_PREFIX = "cifar100_r18_sgd_detarpack_s"


def ci95(xs):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) < 2:
        return (float(xs.mean()) if len(xs) else float("nan")), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), 0.0, float(d.mean()), float(d.mean())
    m = float(d.mean())
    h = float(stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return m, h, m - h, m + h


def load(prefix, col):
    """per-seed late-window mean of col for all runs matching prefix.

    R40 exact-match fix (same rule as analyze_rankkx_interp.late_stats): the
    old trailing-'*' glob also swallowed sibling families — e.g. the SGD
    prefix matched the '_spec' spectrum-diagnostic reruns, which silently
    OVERWROTE seeds 0/1/2 in the dict (sorted glob visits them last),
    corrupting sgd_m and every paired_delta_sgd. Dir names must now be
    exactly prefix + digits.
    """
    pat = re.compile(re.escape(prefix) + r"(\d+)")
    out = {}
    for d in sorted(Path("runs").iterdir()):
        m = pat.fullmatch(d.name)
        if m is None or not d.is_dir() or not (d / "metrics.csv").exists():
            continue
        df = pd.read_csv(d / "metrics.csv")
        if col in df.columns:
            vals = df[df["step"] >= LATE_STEP][col].dropna()
            if len(vals):
                out[int(m.group(1))] = float(vals.mean())
    return out


def logslope(x, y):
    """OLS slope in log-log (nan-safe)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main():
    setup(scale=1.2081)  # R30: source canvas ~1.22x final size (report()-measured)
    global LATE_STEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--late-step", type=int, default=LATE_STEP)
    args = ap.parse_args()
    LATE_STEP = args.late_step

    sgd_lmax = load(SGD_PREFIX, "lambda_max")
    sgd_m, sgd_h = ci95(list(sgd_lmax.values()))
    print(f"SGD baseline (n={len(sgd_lmax)}): late ref λmax "
          f"{sgd_m:.2f} ± {sgd_h:.2f}")

    rows = []
    per_arm = {"sam": {}, "linsam": {}}
    for arm, rhos in ARM_RHOS.items():
        for r, prefix in sorted(rhos.items()):
            lm = load(prefix, "lambda_max")
            if not lm:
                print(f"[missing] {arm} ρ={r}")
                continue
            tr = load(prefix, "taylor_res_rel")
            dl = load(prefix, "taylor_d_lin")
            dt = load(prefix, "taylor_d_total")
            lm_m, lm_h = ci95(list(lm.values()))
            shared = sorted(set(lm) & set(sgd_lmax))
            d = np.array([lm[s] for s in shared]) - np.array([sgd_lmax[s]
                                                              for s in shared])
            pm, ph, plo, phi = paired_ci(d)
            row = dict(arm=arm, rho=r, n_seeds=len(lm),
                       lmax_mean=lm_m, lmax_ci95=lm_h,
                       paired_delta_sgd=pm if len(shared) > 1 else float("nan"),
                       paired_ci90_lo=plo if len(shared) > 1 else float("nan"),
                       paired_ci90_hi=phi if len(shared) > 1 else float("nan"))
            for name, vals in (("taylor_res_rel", tr), ("taylor_d_lin", dl),
                               ("taylor_d_total", dt)):
                m, h = ci95(list(vals.values()))
                row[f"{name}_mean"], row[f"{name}_ci95"] = m, h
            rows.append(row)
            per_arm[arm][r] = row
            tr_s = f" res_rel {row['taylor_res_rel_mean']:.4f} d_lin {row['taylor_d_lin_mean']:.4f}" \
                if not np.isnan(row["taylor_res_rel_mean"]) else ""
            print(f"{arm:7s} ρ={r:<6} n={len(lm)}  late λmax {lm_m:6.2f} ± {lm_h:4.2f}"
                  f"  ΔvsSGD {pm:+.2f} [{plo:+.2f},{phi:+.2f}]{tr_s}")

    out = pd.DataFrame(rows)
    out.to_csv("figures/rho_sweep_summary.csv", index=False)
    print("=> figures/rho_sweep_summary.csv")

    # scaling slopes (log-log) of the Taylor quantities in ρ
    for arm in ("sam", "linsam"):
        rs = sorted(per_arm[arm])
        for q in ("taylor_res_rel", "taylor_d_lin"):
            ys = [per_arm[arm][r][f"{q}_mean"] for r in rs]
            print(f"log-log slope {arm} {q}: {logslope(rs, ys):+.2f} "
                  f"(1.0 = linear-in-ρ, 2.0 = quadratic)")

    # ---- figure ----
    # R30 style (layout frozen — stroke/color/title only): SAM family = teal,
    # linSAM = purple; within-family quantities (res_rel vs d_lin) separated
    # by the EXISTING marker/linestyle channel, never by new hues.  SGD and
    # 2/eta references = neutral theory gray.  Legends frameless.
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    colors = {"sam": C["sam"], "linsam": C["linsam"]}
    for arm in ("sam", "linsam"):
        rs = sorted(per_arm[arm])
        if not rs:
            continue
        m = [per_arm[arm][r]["lmax_mean"] for r in rs]
        h = [per_arm[arm][r]["lmax_ci95"] for r in rs]
        axes[0].errorbar(rs, m, yerr=h, marker="o", lw=ST["primary"],
                         color=colors[arm], label=arm)
        tr = [per_arm[arm][r]["taylor_res_rel_mean"] for r in rs]
        dl = [per_arm[arm][r]["taylor_d_lin_mean"] for r in rs]
        if not all(np.isnan(tr)):
            axes[1].loglog(rs, tr, marker="o", lw=ST["primary"],
                           color=colors[arm], label=f"{arm} res_rel")
        if not all(np.isnan(dl)):
            axes[1].loglog(rs, dl, marker="s", lw=ST["secondary"], ls="--",
                           color=colors[arm], alpha=0.7, label=f"{arm} d_lin")
    axes[0].axhline(sgd_m, color=THEORY["light"], ls=":", lw=ST["theory"])
    # R40: xy x=0.98 was DATA coords on the log ρ-axis (range ~0.005–0.1), so
    # the annotation point sat far outside the axes and the default
    # annotation_clip hid the tag entirely (old PDF text layer carries no
    # "SGD" string). Anchor x in AXES fraction, y in data units, so the tag
    # renders at the right edge just above the SGD rule.
    axes[0].annotate(f"SGD {sgd_m:.1f}", (0.98, sgd_m * 1.03), fontsize=9.67,
                     ha="right", color=THEORY["dark"],
                     xycoords=mtransforms.blended_transform_factory(
                         axes[0].transAxes, axes[0].transData))
    axes[0].axhline(GD, color=THEORY["dark"], ls=":", lw=ST["theory"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\rho$"); axes[0].set_ylabel(r"late reference $\lambda_{\mathrm{ref}}$")
    axes[0].set_title(r"\textbf{(a)} sharpness separation vs $\rho$", color="#1A1D21")
    axes[1].set_xlabel(r"$\rho$"); axes[1].set_ylabel("normalized magnitude")
    axes[1].set_title(r"\textbf{(b)} Taylor content vs $\rho$", color="#1A1D21")
    for ax in axes:
        ax.grid(alpha=0.35, lw=0.4, which="both", color=THEORY["light"])
        ax.legend()
    # No in-figure suptitle: the paper caption carries the framing
    # ("ρ-sweep, full SAM vs. linearized SAM") — same rule as fig 26.
    fig.tight_layout()
    # R40 atomic publish (tmp + rename): a paper build can never pick up a
    # half-written PDF from figures/.
    tmp = Path("figures/44_rho_sweep.tmp_atomic.pdf")
    fig.savefig(tmp, bbox_inches="tight")
    os.replace(tmp, Path("figures/44_rho_sweep.pdf"))
    print("=> figures/44_rho_sweep.pdf")


if __name__ == "__main__":
    main()
