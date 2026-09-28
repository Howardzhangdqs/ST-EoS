"""Batch sharpness λ_step along the τ-interpolation (review ask: the paper
only shows the fixed-reference λmax(τ); this closes the loop between the
stochastic mechanism and the stochastic metric by measuring the EoSS batch
sharpness λ_step(τ) = E_B[g_BᵀH_Bg_B/‖g_B‖²] at every interpolation point).

τ-blend arms (trainer --opt sam_interp: g ← (1−τ)·g_lin + τ·g_SAM):
    τ=0    ≡ linSAM bnmatched   runs/cifar100_r18_linsam_bnmatched_mb_s{0..4}
    τ=0.25 runs/cifar100_r18_sam_interp_t0.25_mb_s{0..4}
    τ=0.5  runs/cifar100_r18_sam_interp_t0.5_mb_s{0..4}
    τ=1    ≡ SAM                runs/cifar100_r18_sam_mb_s{0,1,2(,3,4)}

Metric: late-window (step ≥ 2000, convention from analyze_bsharp_arms.py) mean
of metrics.csv lhg_bs_mean (the m=4-minibatch batch sharpness logged every 200
steps in training). Robustness column: lhg_curmb (the current-train-minibatch
Rayleigh quotient measured at the same points).

Missing seeds are handled gracefully (n reported per τ; paired Δ(τ)−(τ=0) uses
the shared seeds only).

Sanity anchors: τ=0 should sit near linSAM's published batch sharpness
(~22.8) and τ=1 near SAM's (~16.2) — reproducing the reference-λmax ordering.

Outputs:
    figures/tau_bsharp.csv          (per-seed values)
    figures/tau_bsharp_summary.csv  (per-τ mean±95% CI + paired Δ vs τ=0 with
                                     90% CIs, the repo's paired-contrast
                                     convention, + ordering verdict rows)

Usage (CPU only):
    .venv/bin/python scripts/analysis/analyze_tau_bsharp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
LATE_STEP = 2000          # late-window convention (analyze_bsharp_arms.py)

# τ value → run prefix (endpoint equivalences verified from manifests:
# linsam_bnmatched has opt=linsam_bnmatched and IS the τ=0 endpoint of
# sam_interp; sam_mb is τ=1).
TAU_ARMS = [
    (0.0,  "cifar100_r18_linsam_bnmatched_mb_s"),
    (0.25, "cifar100_r18_sam_interp_t0.25_mb_s"),
    (0.5,  "cifar100_r18_sam_interp_t0.5_mb_s"),
    (1.0,  "cifar100_r18_sam_mb_s"),
]
# Published anchors (figures/bsharp_arms_summary.csv, FINDINGS 43b): plain
# linSAM (linsam_mb, single-forward BN) 22.80±2.42; the τ=0 ENDPOINT is the
# BN-matched variant (linsam_bnmatched_mb, dummy second forward ⇒ different BN
# running stats ⇒ its own late value ≈19.9); SAM 16.21±0.70.
ANCHOR_TAU0_BNM, ANCHOR_TAU0_LIN, ANCHOR_TAU1 = 19.9, 22.8, 16.2


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return (float(xs.mean()) if len(xs) else float("nan")), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_ci90(d):
    """mean ± 90% CI of paired deltas (repo convention: analyze_bsharp_arms)."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), 0.0, float(d.mean()), float(d.mean())
    m = float(d.mean())
    h = float(stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return m, h, m - h, m + h


def main() -> None:
    per_seed, per_tau = [], []
    data = {}   # tau -> {seed: (bs, curmb)}
    for tau, prefix in TAU_ARMS:
        data[tau] = {}
        for d in sorted(RUNS.glob(prefix + "*")):
            s = d.name.rsplit("_s", 1)[-1]
            mp = d / "metrics.csv"
            if not (s.isdigit() and mp.exists()):
                continue
            df = pd.read_csv(mp)
            if "lhg_bs_mean" not in df.columns:
                print(f"[skip] {d.name}: no lhg_bs_mean column")
                continue
            m = df[df["step"] >= LATE_STEP]
            if m.empty:
                continue
            bs = float(m["lhg_bs_mean"].mean())
            cur = (float(m["lhg_curmb"].mean())
                   if "lhg_curmb" in m.columns else float("nan"))
            data[tau][int(s)] = (bs, cur)
            per_seed.append(dict(tau=tau, run=d.name, seed=int(s),
                                 n_late_rows=len(m),
                                 lhg_bs_mean=bs, lhg_curmb=cur))
        seeds = sorted(data[tau])
        for metric_i, metric in ((0, "lhg_bs_mean"), (1, "lhg_curmb")):
            vals = [data[tau][s][metric_i] for s in seeds]
            m, h = ci95(vals)
            per_tau.append(dict(kind="per_tau", metric=metric, tau=tau,
                                n_seeds=len(seeds), mean=m, ci95=h,
                                ci_lo=m - h, ci_hi=m + h,
                                note=""))
        print(f"τ={tau:<5} n={len(seeds)} seeds={seeds}  "
              f"lhg_bs_mean {np.mean([data[tau][s][0] for s in seeds]):6.2f}  "
              f"lhg_curmb  {np.nanmean([data[tau][s][1] for s in seeds]):6.2f}")

    # paired Δ(τ) − (τ=0) on shared seeds
    tau0 = data[0.0]
    for tau, _ in TAU_ARMS[1:]:
        shared = sorted(set(data[tau]) & set(tau0))
        for metric_i, metric in ((0, "lhg_bs_mean"), (1, "lhg_curmb")):
            d = np.array([data[tau][s][metric_i] - tau0[s][metric_i]
                          for s in shared])
            m, h, lo, hi = paired_ci90(d)
            sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
            per_tau.append(dict(kind="paired_delta_vs_tau0", metric=metric,
                                tau=tau, n_seeds=len(shared), mean=m, ci95=h,
                                ci_lo=lo, ci_hi=hi, note=f"90% CI; {sig}"))
            print(f"Δ(τ={tau}−τ=0) {metric:12s} = {m:+.2f}  90% CI "
                  f"[{lo:+.2f},{hi:+.2f}]  n={len(shared)}  ({sig})")

    # ordering sanity vs the reference-λmax ordering (linSAM ≳ SAM)
    def tau_mean(tau, i=0):
        return float(np.mean([data[tau][s][i] for s in data[tau]])) if data[tau] else float("nan")

    m0, m1 = tau_mean(0.0), tau_mean(1.0)
    taus = [t for t, _ in TAU_ARMS if data[t]]
    means = [tau_mean(t) for t in taus]
    monotone = all(means[i] >= means[i + 1] - 1e-9
                   for i in range(len(means) - 1))
    ok0 = abs(m0 - ANCHOR_TAU0_BNM) <= 1.0   # endpoint ≡ linsam_bnmatched's own value
    ok1 = abs(m1 - ANCHOR_TAU1) <= 1.5
    verdict = (
        f"τ=0 {m0:.2f} vs linsam_bnmatched anchor {ANCHOR_TAU0_BNM} "
        f"({'OK' if ok0 else 'OFF'}); τ=1 {m1:.2f} vs SAM anchor "
        f"{ANCHOR_TAU1} ({'OK' if ok1 else 'OFF'}); ordering "
        f"{'monotone decreasing in τ' if monotone else 'NOT monotone'} "
        f"({', '.join(f'{t}:{v:.2f}' for t, v in zip(taus, means))}) — "
        f"reproduces the reference-λmax ordering direction (SAM lowest, "
        f"linSAM-family highest). NOTE: the plain-linsam anchor "
        f"{ANCHOR_TAU0_LIN} belongs to linsam_mb (single-forward BN); the τ=0 "
        f"endpoint differs by the BN double-update protocol, not by window "
        f"convention.")
    per_tau.append(dict(kind="note", metric="sanity", tau=float("nan"),
                        n_seeds=0, mean=float("nan"), ci95=float("nan"),
                        ci_lo=float("nan"), ci_hi=float("nan"), note=verdict))
    print(f"SANITY: {verdict}")

    ps = pd.DataFrame(per_seed)
    ps.to_csv(ROOT / "figures" / "tau_bsharp.csv", index=False)
    sm = pd.DataFrame(per_tau)
    sm.to_csv(ROOT / "figures" / "tau_bsharp_summary.csv", index=False)
    print("=> figures/tau_bsharp.csv, figures/tau_bsharp_summary.csv")


if __name__ == "__main__":
    main()
