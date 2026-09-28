"""ViT-tiny EoS-band control: n=3→n=5 aggregate analysis.

Reproduces and extends the published ViT EoS-band numbers from raw runs
(runs/cifar100_vittiny_eos_{sgd,rankkx_d1,linsam,sam}_s{0..4}; s0-2 from
queues r10c/r10d, s3-4 from r10e, all under the identical deterministic
protocol — linsam via the chunked HVP path throughout).

ESTIMATOR (recovered by exhaustive convention search; the unique
combination that reproduces every published n=3 statistic to ±0.05):

    per-seed late statistic : median of metrics.csv:lambda_max over
                              outcome rows with step >= LATE_STEP (=2500)
                              (row set {2600,...,4800,4999}; identical to
                              "last 13 rows" on this outcome grid)
    cross-seed arm level    : MEAN of the per-seed run-medians
                              (the paper's "late medians" 23.7/38.4)
    paired deltas           : per-seed difference of run-medians;
                              point = mean, CI = 90% Student-t on the
                              per-seed differences (ddof=1)
    TOST margin             : 0.25 * |mean delta(SAM-SGD)|  (prespecified
                              family, same as the ResNet arms)

The reproduction gate (run automatically when --seeds is exactly 0,1,2)
compares against the published n=3 values, which live in GATE_N3 purely
as comparison targets — no gate number enters any computation above.
Also reported as diagnostics, because the published appendix phrase
"transient peaks up to 2.3x the SGD peak" does NOT reproduce under the
literal per-run-peak ratio (n=3 max 1.49x); it reproduces only as
peak/(same-seed SGD late median) (2.27x) or peak/(2/eta) (2.34x):

    peak_ratio_late  = run peak / same-seed SGD late-median   -> 2.27 max
    peak_ratio_bound = run peak / (2/eta)                     -> 2.34 max

Outputs:
    figures/vit_eos_band_summary.csv   (per-run / per-seed-pair / summary)
Usage:
    .venv/bin/python scripts/analysis/analyze_vit_eos_band.py              # n=5
    .venv/bin/python scripts/analysis/analyze_vit_eos_band.py --seeds 0,1,2  # gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- constants
RUN_PREFIX = "cifar100_vittiny_eos_{arm}_s{seed}"   # runs/<name>/
ARMS = ["sgd", "rankkx_d1", "linsam", "sam"]
VALUE_COL = "lambda_max"       # canonical fixed-512 ARPACK reference λmax
LATE_STEP = 2500               # late window, step >= (see docstring)
TWO_OVER_ETA = 40.0            # 2/η at the bs512/lr0.05 operating point
CI_LEVEL = 0.90                # paired-delta CI level (90%, as published)
TOST_FRAC = 0.25               # margin = TOST_FRAC * |mean Δ(SAM−SGD)|
DEFAULT_SEEDS = "0,1,2,3,4"
GATE_SEEDS = (0, 1, 2)

PAIRS = [  # (label, arm_a, arm_b): delta = a - b
    ("SAM-SGD", "sam", "sgd"),
    ("linSAM-SAM", "linsam", "sam"),
    ("rankkx-SGD", "rankkx_d1", "sgd"),
]

# Published n=3 values (paper main §5.4 + appendix "ViT control"), used ONLY
# by the reproduction gate comparison table; never in the analysis itself.
GATE_N3 = {
    "arm_level": {"sam": 23.7, "sgd": 38.4},
    "SAM-SGD": (-14.6, -23.4, -5.9),        # (point, ci_lo, ci_hi)
    "linSAM-SAM": (1.5, 0.6, 2.4),
    "rankkx-SGD": (6.4, -0.8, 13.7),
    "sgd_seed_ratios": (0.95, 1.08, 0.84),
    "tost_margin": 3.7,
    "acc": {"sam": 0.370, "sgd": 0.397},
    "peak_word": 2.3,
}


# ------------------------------------------------------------------- helpers
def t_crit(n: int) -> float:
    """Two-sided Student-t critical value at CI_LEVEL, df = n-1."""
    return float(stats.t.ppf(0.5 + CI_LEVEL / 2.0, n - 1))


def paired_ci(d: np.ndarray) -> tuple[float, float, float, float]:
    """Mean of per-seed deltas +/- 90% Student-t CI (ddof=1)."""
    pt = float(d.mean())
    h = t_crit(len(d)) * float(d.std(ddof=1)) / np.sqrt(len(d))
    return pt, pt - h, pt + h, h


def load_run(arm: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(ROOT / "runs" / RUN_PREFIX.format(arm=arm, seed=seed)
                       / "metrics.csv")


def manifest_of(arm: str, seed: int) -> dict:
    mf = ROOT / "runs" / RUN_PREFIX.format(arm=arm, seed=seed) / "manifest.json"
    return json.loads(mf.read_text()) if mf.exists() else {}


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help=f"comma-separated seed list (default {DEFAULT_SEEDS})")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    n = len(seeds)

    # ---- per-run table -------------------------------------------------
    runs, protos = [], set()
    for arm in ARMS:
        for s in seeds:
            df = load_run(arm, s)
            late = df.loc[df["step"] >= LATE_STEP, VALUE_COL].dropna()
            man = manifest_of(arm, s)
            protos.add((man.get("steps"), man.get("lr"), man.get("bs"),
                        man.get("eval_batch"), man.get("outcome_every")))
            runs.append(dict(
                arm=arm, seed=s,
                late_median=float(late.median()),
                late_mean=float(late.mean()),
                ratio_boundary=float(late.median()) / TWO_OVER_ETA,
                peak=float(df[VALUE_COL].max()),
                final_acc=man.get("final_acc"),
                final_acc_full=man.get("final_acc_full"),
            ))
    R = pd.DataFrame(runs)
    sgd_med = dict(zip(R.loc[R.arm == "sgd", "seed"],
                       R.loc[R.arm == "sgd", "late_median"]))
    sgd_pk = dict(zip(R.loc[R.arm == "sgd", "seed"], R.loc[R.arm == "sgd", "peak"]))
    R["peak_ratio_sgdpeak"] = [
        r.peak / sgd_pk[r.seed] for r in R.itertuples()]
    R["peak_ratio_sgdlater"] = [  # diagnostic: reproduces the "2.3×" wording
        r.peak / sgd_med[r.seed] for r in R.itertuples()]
    R["peak_ratio_boundary"] = R.peak / TWO_OVER_ETA
    if len(protos) > 1:
        print(f"WARN protocol drift across runs: {protos}")

    # ---- arm levels ------------------------------------------------------
    print(f"=== ViT EoS-band (bs512 lr0.05, late = median of {VALUE_COL} "
          f"at step>={LATE_STEP})  seeds={seeds} n={n} ===")
    print(f"{'arm':>10} {'n':>2} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}"
          f" {'ratios mean±sd':>16}")
    for arm in ARMS:
        v = R.loc[R.arm == arm, "late_median"].to_numpy()
        rr = R.loc[R.arm == arm, "ratio_boundary"].to_numpy()
        print(f"{arm:>10} {len(v):>2} {v.mean():>8.2f} {np.median(v):>8.2f} "
              f"{v.min():>8.2f} {v.max():>8.2f}   {rr.mean():.3f}±{rr.std(ddof=1):.3f}")
    sgd_rr = R.loc[R.arm == "sgd", "ratio_boundary"].to_numpy()
    print(f"  SGD per-seed late-median/(2/η): "
          + " ".join(f"{x:.3f}" for x in sgd_rr)
          + f"  (mean {sgd_rr.mean():.3f})")

    # ---- paired deltas ---------------------------------------------------
    print(f"\n--- paired deltas (per-seed Δ = run-median difference; "
          f"mean ± {int(CI_LEVEL*100)}% Student-t CI, t({n-1})={t_crit(n):.3f}) ---")
    print(f"{'pair':>12} {'n':>2} {'mean':>8} {'median':>8} {'ci_lo':>9} {'ci_hi':>9}"
          f" {'seeds>0':>8} {'seeds<0':>8}")
    pair_rows, pair_seed_rows = [], []
    for label, a, b in PAIRS:
        va = R.loc[R.arm == a].set_index("seed")["late_median"]
        vb = R.loc[R.arm == b].set_index("seed")["late_median"]
        d = np.array([va[s] - vb[s] for s in seeds])
        pt, lo, hi, h = paired_ci(d)
        print(f"{label:>12} {n:>2} {pt:>+8.2f} {np.median(d):>+8.2f} "
              f"{lo:>+9.2f} {hi:>+9.2f} {int((d>0).sum()):>8} {int((d<0).sum()):>8}")
        print(f"{'':>12} per-seed: "
              + " ".join(f"s{s}={x:+.2f}" for s, x in zip(seeds, d)))
        pair_rows.append(dict(pair=label, n=n, mean=pt, median=float(np.median(d)),
                              ci_lo=lo, ci_hi=hi, half_width=h,
                              t_crit=t_crit(n), seeds_pos=int((d > 0).sum()),
                              seeds_neg=int((d < 0).sum())))
        for s, x in zip(seeds, d):
            pair_seed_rows.append(dict(pair=label, seed=s, delta=x))
    P = pd.DataFrame(pair_rows).set_index("pair")

    # ---- TOST (linSAM vs SAM, prespecified quarter-SAM margin) -----------
    m = TOST_FRAC * abs(P.loc["SAM-SGD", "mean"])
    lo, hi = P.loc["linSAM-SAM", "ci_lo"], P.loc["linSAM-SAM", "ci_hi"]
    tost_pass = bool(lo > -m and hi < m)
    print(f"\nTOST linSAM−SAM: margin = {TOST_FRAC}·|Δ(SAM−SGD)| = {m:.3f}"
          f"  CI [{lo:+.2f},{hi:+.2f}] -> {'PASS (equivalent)' if tost_pass else 'FAIL'}")

    # ---- transient peaks -------------------------------------------------
    print("\n--- transient peaks (max λmax per run) ---")
    for arm in ARMS:
        sub = R.loc[R.arm == arm]
        print(f"{arm:>10}: " + " ".join(
            f"s{int(r.seed)}={r.peak:.1f}" for r in sub.itertuples()))
    rk = R.loc[R.arm == "rankkx_d1"]
    print(f"  rankkx_peak/SGD_peak (literal wording): "
          + " ".join(f"{x:.2f}" for x in rk.peak_ratio_sgdpeak)
          + f"  max {rk.peak_ratio_sgdpeak.max():.2f}×")
    print(f"  rankkx_peak/SGD_late-median (n=3 '2.3×' provenance): "
          + " ".join(f"{x:.2f}" for x in rk.peak_ratio_sgdlater)
          + f"  max {rk.peak_ratio_sgdlater.max():.2f}×")
    print(f"  rankkx_peak/(2/η): "
          + " ".join(f"{x:.2f}" for x in rk.peak_ratio_boundary)
          + f"  max {rk.peak_ratio_boundary.max():.2f}×")

    # ---- accuracies --------------------------------------------------------
    print("\n--- final accuracy (manifest) ---")
    for arm in ARMS:
        sub = R.loc[R.arm == arm]
        fa = sub.final_acc.dropna()
        ff = sub.final_acc_full.dropna()
        line = f"{arm:>10}: final_acc mean {fa.mean():.4f} (n={len(fa)})"
        if len(ff):
            line += f" | final_acc_full mean {ff.mean():.4f} (n={len(ff)})"
        print(line)

    # ---- stability judgments (n=5 focus) ----------------------------------
    lo1, hi1 = P.loc["SAM-SGD", "ci_lo"], P.loc["SAM-SGD", "ci_hi"]
    lo3, hi3 = P.loc["rankkx-SGD", "ci_lo"], P.loc["rankkx-SGD", "ci_hi"]
    print("\n--- stability vs n=3 ---")
    print(f"  Δ(SAM−SGD) CI all-negative: {'YES' if hi1 < 0 else 'NO'} "
          f"([{lo1:+.2f},{hi1:+.2f}])")
    print(f"  linSAM TOST at quarter-SAM margin: {'PASS' if tost_pass else 'FAIL'}")
    print(f"  rankkx drift: seeds>0 {int(P.loc['rankkx-SGD','seeds_pos'])}/{n}, "
          f"CI {'excludes' if (lo3 > 0 or hi3 < 0) else 'contains'} 0 "
          f"([{lo3:+.2f},{hi3:+.2f}])")

    # ---- reproduction gate (only when seeds == 0,1,2) ---------------------
    if tuple(seeds) == GATE_SEEDS:
        print("\n=== reproduction gate (published n=3 vs recomputed) ===")
        n_ok = n_tot = 0

        def chk(name: str, got: float, tgt: float) -> None:
            nonlocal n_ok, n_tot
            ok = abs(got - tgt) <= 0.05
            n_ok, n_tot = n_ok + ok, n_tot + 1
            print(f"  [{'OK' if ok else 'MISS'}] {name:<28} got {got:+9.3f}"
                  f"  published {tgt:+8.2f}")

        for arm, tgt in GATE_N3["arm_level"].items():
            chk(f"arm level {arm}",
                float(R.loc[R.arm == arm, "late_median"].mean()), tgt)
        for label in ("SAM-SGD", "linSAM-SAM", "rankkx-SGD"):
            pt, lo_, hi_ = GATE_N3[label]
            chk(f"Δ({label}) point", float(P.loc[label, "mean"]), pt)
            chk(f"Δ({label}) ci_lo", float(P.loc[label, "ci_lo"]), lo_)
            chk(f"Δ({label}) ci_hi", float(P.loc[label, "ci_hi"]), hi_)
        chk("TOST margin", m, GATE_N3["tost_margin"])
        for i, x in enumerate(sgd_rr):
            chk(f"SGD seed ratio s{i}", float(x), GATE_N3["sgd_seed_ratios"][i])
        for arm, tgt in GATE_N3["acc"].items():
            chk(f"final_acc {arm}",
                float(R.loc[R.arm == arm, "final_acc"].dropna().mean()), tgt)
        print(f"  gate: {n_ok}/{n_tot} within ±0.05"
              + ("  -> ALL HIT" if n_ok == n_tot else "  -> GATE FAILED"))
        print(f"  (wording check, not counted) peak/SGD_peak max "
              f"{rk.peak_ratio_sgdpeak.max():.2f}× vs published "
              f"'up to {GATE_N3['peak_word']}×' — literal ratio does not "
              f"reproduce; peak/SGD-late-median max "
              f"{rk.peak_ratio_sgdlater.max():.2f}× does")

    # ---- CSV ---------------------------------------------------------------
    R["kind"] = "run"
    R["pair"] = ""
    PS = pd.DataFrame(pair_seed_rows)
    PS["kind"] = "pair_seed"
    S = P.reset_index()
    S["kind"] = "pair_summary"
    T = pd.DataFrame([dict(kind="tost", pair="linSAM-SAM", n=n,
                           mean=P.loc["linSAM-SAM", "mean"],
                           median=P.loc["linSAM-SAM", "median"],
                           ci_lo=lo, ci_hi=hi, half_width=m,
                           seeds_pos=int((pd.DataFrame(pair_seed_rows)
                                          .query("pair=='linSAM-SAM'").delta > 0).sum()),
                           seeds_neg=int((pd.DataFrame(pair_seed_rows)
                                          .query("pair=='linSAM-SAM'").delta < 0).sum()))])
    cols = ["kind", "arm", "seed", "pair", "late_median", "late_mean",
            "ratio_boundary", "peak", "peak_ratio_sgdpeak",
            "peak_ratio_sgdlater", "peak_ratio_boundary", "final_acc",
            "final_acc_full", "n", "mean", "median", "ci_lo", "ci_hi",
            "half_width", "t_crit", "seeds_pos", "seeds_neg"]
    out = pd.concat([R, PS, S, T], ignore_index=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols].to_csv(ROOT / "figures" / "vit_eos_band_summary.csv",
                     index=False, float_format="%.4f")
    print(f"\nwrote figures/vit_eos_band_summary.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
