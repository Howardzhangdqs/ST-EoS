"""TOST equivalence tests for the at-scale null results — review round 2.

Upgrading "95% CIs overlap ⇒ statistically indistinguishable" to formal
equivalence evidence (reviewer: absence of significance ≠ evidence of
equivalence). We use the standard two-one-sided-tests (TOST) procedure on
PAIRED per-seed differences: every arm is run under deterministic CUDA with
the same data-stream seed, so seed s of arm A saw the identical batch
sequence as seed s of SGD — the pairing removes the cross-seed variance that
dominates the unpaired comparison.

Procedure (prespecified):
  * endpoint: late-window (step ≥ 2000) mean reference λ_max per seed
    (canonical `_detarpack` protocol, ARPACK k=3 tol 1e-6 every 100).
  * margin:  m = 0.25 · |SGD − SAM| (a quarter of SAM's own reduction —
    i.e. we ask whether an arm's effect is smaller than a quarter of the
    effect the global amplifying action achieves).
  * TOST: reject H0 (|Δ| ≥ m) iff the 90% CI of the paired mean difference
    lies inside (−m, +m).  CI = Δ̄ ± t_{0.95,n−1} · s/√n.

Usage:
    python scripts/analysis/analyze_tost_equivalence.py
    python scripts/analysis/analyze_tost_equivalence.py --arms rankkx_d2 rankkx_d5 ...
Outputs: printed table + figures/tost_equivalence.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LATE_STEP = 2000

# canonical arms (run-dir glob prefix, display name); SGD is the paired
# reference; SAM defines the margin
CANONICAL = {
    "sgd":    "cifar100_r18_sgd_detarpack_s",
    "rank1":  "cifar100_r18_rank1_detarpack_s",
    "sam":    "cifar100_r18_sam_detarpack_s",
    "amp0.5": "cifar100_r18_amp0.5_detarpack_s",
    "amp1.0": "cifar100_r18_amp1.0_detarpack_s",
    # accurate ARPACK-sensing rank-1 damping plant (same edit as rankkx_d1;
    # outcome every 200 instead of 100)
    "truelmax": "cifar100_r18_truelmax_s",
    # review-round-2 rank-k family (as the queue lands)
    "rankkx_d1":     "cifar100_r18_rankkx_d1_s",
    "rankkx_d1_curb": "cifar100_r18_rankkx_d1_curb_s",
    "rankkx_d2":     "cifar100_r18_rankkx_d2_s",
    "rankkx_d5":     "cifar100_r18_rankkx_d5_s",
    "rankkx_d10":    "cifar100_r18_rankkx_d10_s",
    "rankkx_d20":    "cifar100_r18_rankkx_d20_s",
    "rankkx_a5":     "cifar100_r18_rankkx_a5_s",
    "rankkx_a20":    "cifar100_r18_rankkx_a20_s",
}

# 200-epoch standard-recipe arms (external-review upgrade): endpoint is the
# steady-window (epochs 30-120) mean reference lambda per seed; the same
# prespecified margin rule applies (0.25 * |SGD - SAM|).
RECIPE_200EP = {
    "sgd":         "cifar100_r18_sgd_200ep_s",
    "rankkx_d1":   "cifar100_r18_rankkx_d1_200ep_s",
    "sam":         "cifar100_r18_sam_200ep_s",
}

EPOCH = 391
STEADY_200EP = (30 * EPOCH, 120 * EPOCH)  # cosine lr 0.088 -> 0.023


def late_mean(run_dir: Path, steady: tuple[int, int] | None = None) -> float | None:
    f = run_dir / "metrics.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if steady is not None:
        df = df[(df["step"] >= steady[0]) & (df["step"] <= steady[1])]
    else:
        df = df[df["step"] >= LATE_STEP]
    if df.empty:
        return None
    return float(df["lambda_max"].mean())


def collect(prefix: str, steady: tuple[int, int] | None = None) -> dict[int, float]:
    out = {}
    for d in sorted(Path("runs").glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if not s.isdigit():
            continue
        v = late_mean(d, steady)
        if v is not None:
            out[int(s)] = v
    return out


def tost(deltas: np.ndarray, margin: float) -> dict:
    n = len(deltas)
    mean = float(deltas.mean())
    if n < 2:
        return dict(n=n, mean_diff=mean, ci_lo=mean, ci_hi=mean,
                    margin=margin, equivalent=False, verdict="n<2 (no test)")
    h = float(stats.t.ppf(0.95, n - 1) * deltas.std(ddof=1) / np.sqrt(n))
    lo, hi = mean - h, mean + h
    equiv = bool(lo > -margin and hi < margin)
    verdict = ("EQUIVALENT (|effect| < margin at α=0.05)"
               if equiv else "NOT established (CI exits margin)")
    return dict(n=n, mean_diff=mean, ci_lo=lo, ci_hi=hi, margin=margin,
                equivalent=equiv, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-frac", type=float, default=0.25,
                    help="margin = margin_frac × |SGD − SAM| late means")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arm names to test (default: all found)")
    ap.add_argument("--recipe", choices=["canonical", "200ep"], default="canonical",
                    help="canonical = 5000-step at-scale protocol (late window); "
                         "200ep = standard 200-epoch recipe (steady epochs 30-120)")
    args = ap.parse_args()

    if args.recipe == "200ep":
        arms_map, steady, out_csv = RECIPE_200EP, STEADY_200EP, \
            "figures/tost_equivalence_200ep.csv"
        endpoint = "steady(e30-120) mean"
    else:
        arms_map, steady, out_csv = CANONICAL, None, "figures/tost_equivalence.csv"
        endpoint = "late(step>=2000) mean"

    data = {name: collect(pfx, steady) for name, pfx in arms_map.items()}
    data = {k: v for k, v in data.items() if v}
    if "sgd" not in data or "sam" not in data:
        raise SystemExit("need at least the sgd + sam reference runs")

    sgd, sam = data["sgd"], data["sam"]
    margin = args.margin_frac * abs(np.mean(list(sgd.values())) -
                                    np.mean(list(sam.values())))
    print(f"[recipe={args.recipe}] reference: SGD {endpoint} "
          f"{np.mean(list(sgd.values())):.2f} (n={len(sgd)}), "
          f"SAM {np.mean(list(sam.values())):.2f} (n={len(sam)})")
    print(f"prespecified margin m = {args.margin_frac:.2f}·|SGD−SAM| = {margin:.3f}\n")

    arms = args.arms or [a for a in data if a not in ("sgd", "sam")]
    rows = []
    for arm in arms:
        if arm not in data:
            print(f"[skip] {arm}: no runs found")
            continue
        seeds = sorted(set(data[arm]) & set(sgd))
        if not seeds:
            print(f"[skip] {arm}: no seed overlap with SGD")
            continue
        deltas = np.array([data[arm][s] - sgd[s] for s in seeds])
        r = tost(deltas, margin)
        rows.append(dict(arm=arm, **r))
        print(f"{arm:15s} Δ(SGD) = {r['mean_diff']:+7.3f}  "
              f"90% CI [{r['ci_lo']:+7.3f}, {r['ci_hi']:+7.3f}]  "
              f"n={r['n']}  m={margin:.2f}  → {r['verdict']}")

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n=> {out_csv}")


if __name__ == "__main__":
    main()
