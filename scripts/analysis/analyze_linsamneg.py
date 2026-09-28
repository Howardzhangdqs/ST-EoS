"""Sign-flipped linearized SAM (linsamneg) — review round 11 control.

linsamneg applies g ← g − (ρ/‖g‖)·H_B g (opt=linsam_neg, ρ=0.05): the SAME
first-order correction magnitude as linSAM but along the NEGATIVE curvature
direction — an anti-SAM descent control. If SAM's effect on sharpness were a
generic "large correction" effect, linsamneg should move λmax/batch sharpness
the same way; if it is direction-specific, linsamneg should sit at/near SGD.

Conventions copied EXACTLY from scripts/analysis/analyze_linsam.py (window step ≥ 2000,
late per-seed mean → cross-seed mean ± 95% CI Student-t ppf 0.975 ddof=1,
paired Δ with 90% CI ppf 0.95) and scripts/analysis/analyze_bsharp_arms.py (TOST at
the pre-specified margin 0.25·|SGD−SAM| on the same quantity, paired t).

Arms:
    linsamneg  cifar100_r18_linsamneg_mb_s{0..4}   (n=5, mb rerun family)
    SGD (λmax) cifar100_r18_sgd_detarpack_s*       — the SAME baseline family
               analyze_linsam.py pairs against (deterministic-ARPACK
               protocol; s{0..4} now exist)
    SAM  (λmax) cifar100_r18_sam_detarpack_s*      — for the TOST margin
    linSAM     cifar100_r18_linsam_mb_s{0..4}      (n=5)
    SGD (bsharp) cifar100_r18_sgd_detarpack_mb2_s* — the only SGD arm with
               the lhg_bs_mean column (the 20.51 published value, n=3)

Reported: late reference λmax mean±CI; paired Δ(linsamneg−SGD) + TOST;
paired Δ(linsamneg−linSAM); late batch sharpness (lhg_bs_mean) ± CI +
paired Δ vs SGD-mb2; the empirical norm-matched check (late mean
taylor_d_lin = ‖g_lin−g‖/‖g‖ — a sign-agnostic magnitude) linsamneg vs
linSAM. Numbers as measured, no spin.

Outputs: figures/linsamneg_summary.csv
Usage:  .venv/bin/python scripts/analysis/analyze_linsamneg.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LATE_STEP = 2000

NEG_PFX = "cifar100_r18_linsamneg_mb_s"
LIN_PFX = "cifar100_r18_linsam_mb_s"
SGD_PFX = "cifar100_r18_sgd_detarpack_s"        # λmax baseline (analyze_linsam)
SAM_PFX = "cifar100_r18_sam_detarpack_s"        # TOST margin anchor
SGD_MB_PFX = "cifar100_r18_sgd_detarpack_mb2_s"  # lhg baseline (20.51, n=3)


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


def tost_p(d, margin):
    """Two one-sided tests p-value for |mean(d)| < margin (paired, t).
    Copied from analyze_bsharp_arms.py."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), False
    se = d.std(ddof=1) / np.sqrt(n)
    if se <= 0:
        return 0.0 if abs(d.mean()) < margin else 1.0, abs(d.mean()) < margin
    df = n - 1
    t_lo = (d.mean() - (-margin)) / se
    t_hi = (d.mean() - margin) / se
    p = max(stats.t.cdf(t_lo, df), 1 - stats.t.cdf(t_hi, df))
    return float(p), bool(p < 0.05)


def load(prefix, col="lambda_max"):
    out = {}
    for d in sorted(Path("runs").glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            df = pd.read_csv(d / "metrics.csv")
            if col in df.columns:
                out[int(s)] = df
    return out


def late_mean(df, col="lambda_max"):
    return float(df[df["step"] >= LATE_STEP][col].dropna().mean())


def main():
    neg = load(NEG_PFX)
    sgd = load(SGD_PFX)
    sam = load(SAM_PFX)
    lin = load(LIN_PFX)
    sgd_mb = load(SGD_MB_PFX, col="lhg_bs_mean")
    if not neg:
        raise SystemExit("no linsamneg runs found")
    print(f"linsamneg seeds {sorted(neg)} | sgd {sorted(sgd)} | "
          f"sam {sorted(sam)} | linsam {sorted(lin)} | sgd_mb {sorted(sgd_mb)}")

    rows = []

    # ---- late reference λmax -------------------------------------------------
    per = {name: {s: late_mean(runs[s]) for s in runs}
           for name, runs in (("linsamneg", neg), ("SGD", sgd),
                              ("SAM", sam), ("linSAM", lin))}
    for name in ("linsamneg", "SGD", "SAM", "linSAM"):
        vals = list(per[name].values())
        m, h = ci95(vals)
        rows.append(dict(quantity="late_lmax", arm=name, mean=m, ci95=h,
                         n=len(vals)))
        print(f"{name:9s} late λmax {m:6.2f} ± {h:4.2f}  (n={len(vals)})")

    # paired Δ(linsamneg−SGD), λmax — SGD family = sgd_detarpack (analyze_linsam)
    shared = sorted(set(neg) & set(sgd))
    d = [late_mean(neg[s]) - late_mean(sgd[s]) for s in shared]
    m, h, lo, hi = paired_ci(d)
    # canonical TOST margin 0.25·|SGD−SAM| on λmax (same quantity)
    margin = 0.25 * abs(np.mean(list(per["SGD"].values()))
                        - np.mean(list(per["SAM"].values())))
    tp, tok = tost_p(d, margin)
    rows.append(dict(quantity="lmax_paired_delta_vs_SGD", arm="linsamneg−SGD",
                     mean=m, ci95=h, n=len(shared), ci_lo=lo, ci_hi=hi,
                     tost_margin=margin, tost_p=tp, tost_pass=tok))
    eq = ("TOST-equivalent at margin" if tok else
          ("SIGNIFICANT difference (90% CI excludes 0)" if lo > 0 or hi < 0
           else "no measurable difference at this power"))
    print(f"Δ(linsamneg−SGD) λmax = {m:+.2f}  90% CI [{lo:+.2f},{hi:+.2f}]  "
          f"n={len(shared)}  margin ±{margin:.2f}  p={tp:.4f}  {eq}")

    # paired Δ(linsamneg−linSAM), λmax (linsam_mb n=5)
    shared = sorted(set(neg) & set(lin))
    d = [late_mean(neg[s]) - late_mean(lin[s]) for s in shared]
    m, h, lo, hi = paired_ci(d)
    rows.append(dict(quantity="lmax_paired_delta_vs_linSAM",
                     arm="linsamneg−linSAM", mean=m, ci95=h, n=len(shared),
                     ci_lo=lo, ci_hi=hi))
    print(f"Δ(linsamneg−linSAM) λmax = {m:+.2f}  90% CI [{lo:+.2f},{hi:+.2f}]  "
          f"n={len(shared)}")

    # ---- late batch sharpness (lhg_bs_mean) -----------------------------------
    per_bs = {name: {s: late_mean(runs[s], "lhg_bs_mean") for s in runs}
              for name, runs in (("linsamneg", neg), ("SGDmb2", sgd_mb),
                                 ("linSAM", lin))}
    for name in ("linsamneg", "SGDmb2", "linSAM"):
        vals = list(per_bs[name].values())
        m, h = ci95(vals)
        rows.append(dict(quantity="late_bsharp", arm=name, mean=m, ci95=h,
                         n=len(vals)))
        print(f"{name:9s} late batch sharpness {m:6.2f} ± {h:4.2f}  (n={len(vals)})")
    shared = sorted(set(neg) & set(sgd_mb))
    d = [late_mean(neg[s], "lhg_bs_mean") - late_mean(sgd_mb[s], "lhg_bs_mean")
         for s in shared]
    m, h, lo, hi = paired_ci(d)
    rows.append(dict(quantity="bsharp_paired_delta_vs_SGDmb2",
                     arm="linsamneg−SGDmb2", mean=m, ci95=h, n=len(shared),
                     ci_lo=lo, ci_hi=hi))
    print(f"Δ(linsamneg−SGDmb2) bsharp = {m:+.2f}  90% CI [{lo:+.2f},{hi:+.2f}]  "
          f"n={len(shared)}")

    # ---- empirical norm-matched check (taylor_d_lin, sign-agnostic) -----------
    for name, runs in (("linsamneg", neg), ("linSAM", lin)):
        vals = [late_mean(runs[s], "taylor_d_lin") for s in runs
                if "taylor_d_lin" in runs[s].columns]
        m, h = ci95(vals)
        rows.append(dict(quantity="taylor_d_lin_rel_late", arm=name, mean=m,
                         ci95=h, n=len(vals)))
        print(f"{name:9s} taylor_d_lin (‖g_lin−g‖/‖g‖) late mean "
              f"{m:.4f} ± {h:.4f}  (n={len(vals)})")

    out = pd.DataFrame(rows)
    out.to_csv("figures/linsamneg_summary.csv", index=False)
    print("=> figures/linsamneg_summary.csv")


if __name__ == "__main__":
    main()
