#!/usr/bin/env python3
"""ROUND 27 -- ordinary paired differences: 90% -> 95% CI recalculation.

Rule under review: the ordinary paired difference now uses a 95% t-CI
(t.ppf(0.975, n-1); resolved95 <=> CI excludes 0); TOST equivalence keeps
its 90% CI (NOT touched here).  Every number is recomputed FROM RAW
per-seed data in runs/ -- never back-derived from the paper's rounded
90% CIs.

Window / statistic conventions replicated exactly from the source scripts:

  group                    per-seed statistic (source script lines)       pairs
  -----------------------  -------------------------------------------  -----
  bsharp central arms      late(step>=2000) mean lhg_bs_mean and          arm - SGD(mb2),
  (S5.1 / Fig.2)           lambda_max; LATE_STEP=2000                     shared seeds
                           [analyze_bsharp_arms.py L37, L48-54, L142-144]
  rank-k Table 9 + D.15    late(step>=2000) mean lambda_max               arm - SGD(detarpack)
                           [analyze_tost_equivalence.py L35, L39-57,      shared seeds
                            L72-83]
  regime sweep Table 8     window mean lambda_max; late = step>=2000,     d1/amp - sgd
  (S5.2 / D.5)             mid = 200<=step<2000; B<=128 late, B>=512 mid  per cell, seeds
                           [analyze_regime_sweep.py L45, L51-72,          (0,1,2)
                            L104, L107-127, L222-229]
  D.6 BN controls          late(step>=2000) mean lambda_max               sgd_bn2x-sgd,
                           [analyze_r7_bn.py L51-65, L95-99]              linsam_bnmatched-
                                                                         linsam, sam_bnfreeze-sam
  D.7 ViT                  MEDIAN lambda_max over step>=2500 (median,     sam-sgd,
                           NOT mean); paired diff of per-seed medians     linsam-sam,
                           [analyze_vit_eos_band.py L57, L64-68,          rankkx_d1-sgd, n=5
                            L121, L167-171]
  D.8 200ep                steady epochs 30-120 (steps 11730..46920,      sam/rankkx_d1 - sgd
                           inclusive) mean lambda_max; complete runs      shared seeds
                           only (last step >= 78200-391)                   [analyze_long_recipe.py
                                                                            L54-63, L105-119]
  D.11 rho sweep           late(step>=2000) mean lambda_max               sam/linsam - sgd
                           [analyze_rho_sweep.py L28-40, L61-72]          per rho
  D.11 tau interpolation   late(step>=2000) mean lhg_bs_mean (lambda_step sam_mb(tau=1) -
                           = the paper's lstep contrast!)                 linsam_bnmatched(tau=0)
                           [analyze_tau_interpolation.py L56, L59,        [L62-67, L107]
                            L62-67]
  D.11 linsamneg           late(step>=2000) means; lambda_max and         linsamneg - SGD /
                           lhg_bs_mean                                    - linSAM / - SGDmb2
                           [analyze_linsamneg.py L41-47, L96-97]

Anchor gate (hard): the recomputed 90% CI must reproduce the published
anchor (rounded to the paper's printing precision) before the 95% numbers
of that group are usable.  Tolerance = half a printing unit + epsilon
(2dp anchors -> 0.0101, 1dp anchors -> 0.0501).

Outputs: markdown tables to stdout + figures/r27_ci95_summary.csv
Usage:  python3 scripts/analysis/analyze_r27_ci95.py
(run from the repo root; runs/ is read-only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT_CSV = ROOT / "figures" / "r27_ci95_summary.csv"

TOL2 = 0.0101   # paper prints 2 decimals -> rounding error <= 0.005
TOL1 = 0.0501   # paper prints 1 decimal  -> rounding error <= 0.05

ROWS: list[dict] = []


# ----------------------------------------------------------------- helpers
def paired(d, level):
    """mean +/- t-CI of paired diffs (ddof=1); degenerate for n<2."""
    d = np.asarray([x for x in d if x == x], float)   # NaN-safe (regime conv.)
    n = len(d)
    m = float(d.mean())
    if n < 2:
        return m, m, m, n
    h = float(stats.t.ppf(1 - (1 - level) / 2, n - 1)
              * d.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def seed_runs(prefix):
    """{seed: run_dir} for runs/<prefix>*/ with a metrics.csv (repo conv.)."""
    out = {}
    for d in sorted(RUNS.glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            out[int(s)] = d
    return out


def window_stat(prefix, col, lo, hi=None, hi_inc=False, how="mean",
                min_last_step=None):
    """Per-seed window statistic of col (repo loaders' exact semantics).

    how='mean'   -> mean over the window (NaN-skipping, as pandas .mean())
    how='median' -> dropna then median (analyze_vit_eos_band.py L121)
    """
    out = {}
    for s, d in seed_runs(prefix).items():
        df = pd.read_csv(d / "metrics.csv")
        if min_last_step is not None and df["step"].iloc[-1] < min_last_step:
            continue
        m = df[df["step"] >= lo]
        if hi is not None:
            m = m[m["step"] <= hi] if hi_inc else m[m["step"] < hi]
        v = m[col]
        if how == "median":
            v = v.dropna()
            if len(v):
                out[s] = float(v.median())
        else:
            if v.notna().any():
                out[s] = float(v.mean())
    return out


def add(group, comparison, table, d_map, window, d, anchor=None,
        note=""):
    """d_map: {seed: arm_value}; d: list of paired diffs (arm - reference).

    anchor: (mean, lo, hi, tol) at the 90% level -- the published value.
    """
    seeds = sorted(d_map)
    m90, lo90, hi90, n = paired(d, 0.90)
    m95, lo95, hi95, _ = paired(d, 0.95)
    resolved = None if n < 2 else bool(lo95 > 0 or hi95 < 0)
    a_txt = a_ok = None
    if anchor is not None:
        am, alo, ahi, tol = anchor
        a_txt = f"{am:+.2f} [{alo:+.2f},{ahi:+.2f}]".replace("+0.", "+.").replace("-0.", "-.")
        a_txt = f"{am:+.2f} [{alo:+.2f},{ahi:+.2f}]"
        a_ok = (abs(m90 - am) <= tol and abs(lo90 - alo) <= tol
                and abs(hi90 - ahi) <= tol)
    ROWS.append(dict(
        group=group, comparison=comparison, table=table,
        n=n, seeds=",".join(str(s) for s in seeds), window=window,
        mean_delta=m90, ci90_lo=lo90, ci90_hi=hi90,
        ci95_lo=lo95, ci95_hi=hi95,
        resolved95=("" if resolved is None else str(resolved)),
        anchor=("" if a_txt is None else a_txt),
        anchor_ok=("" if a_ok is None else str(a_ok)),
        per_seed_deltas=" ".join(f"s{s}:{x:+.3f}"
                                 for s, x in zip(seeds, d)),
        note=note))
    return ROWS[-1]


def deltas(arm, ref):
    shared = sorted(set(arm) & set(ref))
    return shared, [arm[s] - ref[s] for s in shared]


# ------------------------------------------------- 1. bsharp central arms
# analyze_bsharp_arms.py L37 (LATE_STEP=2000), L48-54 (ARMS prefixes),
# L142-144 (per-seed late means of lhg_bs_mean / lambda_max), L186-194
# (paired vs SGD on shared seeds).
def g1_bsharp():
    ARMS = [  # (label, run prefix) -- analyze_bsharp_arms.py L48-54
        ("SGD",         "cifar100_r18_sgd_detarpack_mb2_s"),
        ("rank-1 damp", "cifar100_r18_rankkx_d1_mb_s"),
        ("rank-1 amp",  "cifar100_r18_amp05_mb_s"),
        ("linSAM",      "cifar100_r18_linsam_mb_s"),
        ("SAM",         "cifar100_r18_sam_mb_s"),
    ]
    bs = {l: window_stat(p, "lhg_bs_mean", 2000) for l, p in ARMS}
    lm = {l: window_stat(p, "lambda_max", 2000) for l, p in ARMS}
    for label, _ in ARMS[1:]:
        for per, q in ((bs, "bsharp"), (lm, "lmax_ref")):
            shared, d = deltas(per[label], per["SGD"])
            anchor = None
            if label == "SAM" and q == "bsharp":
                anchor = (-4.30, -5.93, -2.66, TOL2)   # paper S5.1 / Fig.2
            add("bsharp", f"Δ({label}−SGD) {q}", "Fig.2/S5.1",
                {s: per[label][s] - per["SGD"][s] for s in shared},
                "step≥2000 mean", d, anchor)


# --------------------------------------- 2. rank-k family (Table 9, D.15)
# analyze_tost_equivalence.py L35 (LATE_STEP=2000), L39-57 (CANONICAL),
# L72-83 (late_mean = window mean of lambda_max), L151-156 (paired vs sgd).
def g2_rankk():
    CANONICAL = {  # analyze_tost_equivalence.py L39-57
        "sgd":           "cifar100_r18_sgd_detarpack_s",
        "rank1":         "cifar100_r18_rank1_detarpack_s",
        "sam":           "cifar100_r18_sam_detarpack_s",
        "amp0.5":        "cifar100_r18_amp0.5_detarpack_s",
        "amp1.0":        "cifar100_r18_amp1.0_detarpack_s",
        "truelmax":      "cifar100_r18_truelmax_s",
        "rankkx_d1":     "cifar100_r18_rankkx_d1_s",
        "rankkx_d1_curb": "cifar100_r18_rankkx_d1_curb_s",
        "rankkx_d2":     "cifar100_r18_rankkx_d2_s",
        "rankkx_d5":     "cifar100_r18_rankkx_d5_s",
        "rankkx_d10":    "cifar100_r18_rankkx_d10_s",
        "rankkx_d20":    "cifar100_r18_rankkx_d20_s",
        "rankkx_a5":     "cifar100_r18_rankkx_a5_s",
        "rankkx_a20":    "cifar100_r18_rankkx_a20_s",
    }
    ANCHORS = {   # paper Table 9 (tab:interp) / D.15 (tab:curb)
        "rankkx_a20":     (+4.88, +1.74, +8.02, TOL2),
        "rankkx_d10":     (+0.88, +0.15, +1.62, TOL2),  # = TOST CI, kept 90%
        "rankkx_d1_curb": (+1.16, -0.26, +2.58, TOL2),  # D.15
    }
    TABLE9 = ["rankkx_d1", "rankkx_d2", "rankkx_d5", "rankkx_d10",
              "rankkx_d20", "amp1.0", "rankkx_a5", "rankkx_a20"]
    sgd = window_stat(CANONICAL["sgd"], "lambda_max", 2000)
    for arm, pfx in CANONICAL.items():
        if arm in ("sgd", "sam"):
            continue
        per = window_stat(pfx, "lambda_max", 2000)
        if not per:
            continue
        shared, d = deltas(per, sgd)
        add("rankk", f"Δ({arm}−SGD)", "Table 9" if arm in TABLE9 else "extra",
            {s: per[s] - sgd[s] for s in shared}, "step≥2000 mean", d,
            ANCHORS.get(arm))


# --------------------------------------------- 3. regime sweep (Table 8)
# analyze_regime_sweep.py L45 (LATE_STEP), L48 (SEEDS=(0,1,2)), L51-72
# (CELLS + templates; the 128/m09 cell reuses the round-3/5 mb runs),
# L104 (WINDOWS), L107-127 (window mean of lambda_max), L197-203 (paired),
# L222-229 (window selection: B<=128 late, B>=512 mid).
def g3_regime():
    CELLS = [  # (bs, mu, tag, anchor_cell) -- analyze_regime_sweep.py L51-60
        (2048, 0.9, "m09", False), (512, 0.9, "m09", False),
        (128,  0.9, "m09", True),  (32,   0.9, "m09", False),
        (2048, 0.0, "m0",  False), (512,  0.0, "m0",  False),
        (128,  0.0, "m0",  False), (32,   0.0, "m0",  False),
    ]
    ARM_TPL = {"sgd": "cifar100_r18_bs{bs}_{mt}_sgd_s{s}",
               "d1":  "cifar100_r18_bs{bs}_{mt}_d1_s{s}",
               "amp": "cifar100_r18_bs{bs}_{mt}_amp05_s{s}"}
    ANCHOR_TPL = {  # the B=128/mu=0.9 anchor cell reuses the round-3/5 runs
        "sgd": "cifar100_r18_sgd_detarpack_mb2_s{s}",
        "d1":  "cifar100_r18_rankkx_d1_mb_s{s}",
        "amp": "cifar100_r18_amp05_mb_s{s}"}
    WINDOWS = [("late", 2000, None), ("mid", 200, 2000)]
    # paper anchors: Table 8 (tab:regimebridge, d1 arm at mu=0.9) + D.5 text
    # (B=128/mu=0 damping endpoint); all printed at 1 decimal.
    ANCHORS = {
        (0.9, 128,  "d1", "late"): (+0.6,  -2.4,  +3.6,  TOL1),
        (0.9, 512,  "d1", "mid"):  (+12.1, -15.5, +39.7, TOL1),
        (0.9, 2048, "d1", "mid"):  (+44.0, +41.0, +47.0, TOL1),
        (0.0, 128,  "d1", "late"): (+49.7, +13.1, +86.2, TOL1),
    }
    for bs, mu, mt, anchor_cell in CELLS:
        tpl = ANCHOR_TPL if anchor_cell else ARM_TPL
        for wname, lo, hi in WINDOWS:
            per = {}
            for arm in ("sgd", "d1", "amp"):
                pfx = tpl[arm].format(bs=bs, mt=mt, s="")   # ends in "_s"
                per[arm] = window_stat(pfx, "lambda_max", lo, hi)
            for arm in ("d1", "amp"):
                if not per[arm]:
                    continue
                shared, d = deltas(per[arm], per["sgd"])
                if not shared:
                    continue
                wsel = (bs <= 128 and wname == "late") or \
                       (bs >= 512 and wname == "mid")
                key = (mu, bs, arm, wname)
                add("regime", f"Δ({arm}−sgd) B={bs}/μ={mu}",
                    "Table 8" if wsel and mu == 0.9 and arm == "d1" else
                    ("D.5 text" if key in ANCHORS else "extra"),
                    {s: per[arm][s] - per["sgd"][s] for s in shared},
                    f"{wname} window", d,
                    ANCHORS.get(key),
                    note="selected window" if wsel else "")


# ------------------------------------------------- 4. D.6 BN controls
# analyze_r7_bn.py L51 (LATE_STEP), L55-65 (arm prefixes + SGD candidates),
# L95-99 (late mean lambda_max), L216-233 (BN contrasts), L107-110
# (manifest flag match for the SGD baseline family).
def g4_bn():
    R7 = {"sam_bnfreeze":     "cifar100_r18_sam_bnfreeze_mb_s",
          "sgd_bn2x":         "cifar100_r18_sgd_bn2x_mb_s",
          "linsam_bnmatched": "cifar100_r18_linsam_bnmatched_mb_s"}
    BASE = {"sam":    "cifar100_r18_sam_mb_s",
            "linsam": "cifar100_r18_linsam_mb_s"}
    SGD_CANDS = ["cifar100_r18_sgd_detarpack_mb2_s",
                 "cifar100_r18_sgd_detarpack_mb_s"]
    FLAGS = ["bs", "steps", "momentum", "lr", "multibatch", "outcome_arpack",
             "outcome_every", "eval_batch", "schedule"]

    def manifest(prefix, seed=0):
        p = RUNS / f"{prefix}{seed}" / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    def flags_match(a, b):
        return a is not None and b is not None and \
            all(a.get(k) == b.get(k) for k in FLAGS)

    r7_ref = manifest(R7["sam_bnfreeze"]) or manifest(BASE["sam"])
    sgd_pfx = next((c for c in SGD_CANDS
                    if flags_match(manifest(c), r7_ref)), SGD_CANDS[0])
    late = {name: window_stat(p, "lambda_max", 2000)
            for name, p in {**R7, **BASE, "sgd": sgd_pfx}.items()}
    CONTRASTS = [  # (reference, control, paper anchor)  -- D.6
        ("sgd", "sgd_bn2x", (-2.1, -4.4, +0.2, TOL1)),
        ("linsam", "linsam_bnmatched", (-2.8, -4.7, -0.9, TOL1)),
        ("sam", "sam_bnfreeze", None),
    ]
    for ref, ctrl, anchor in CONTRASTS:
        shared, d = deltas(late[ctrl], late[ref])
        add("bn", f"Δ({ctrl}−{ref})", "D.6",
            {s: late[ctrl][s] - late[ref][s] for s in shared},
            "step≥2000 mean", d, anchor)


# ------------------------------------------------------- 5. D.7 ViT
# analyze_vit_eos_band.py L54-61 (prefix/arms/LATE_STEP=2500), L64-68
# (PAIRS), L121 (per-seed MEDIAN of lambda_max over step>=2500),
# L167-171 (paired per-seed median differences).
def g5_vit():
    PFX = "cifar100_vittiny_eos_{arm}_s{seed}"
    per = {arm: window_stat(PFX.format(arm=arm, seed=""), "lambda_max",
                            2500, how="median")
           for arm in ("sgd", "rankkx_d1", "linsam", "sam")}
    PAIRS = [  # analyze_vit_eos_band.py L64-68
        ("SAM−SGD",    "sam",        "sgd",        (-15.1, -18.8, -11.4, TOL1)),
        ("linSAM−SAM", "linsam",     "sam",        (+1.7,  +1.2,  +2.1,  TOL1)),
        ("rankkx−SGD", "rankkx_d1",  "sgd",        (+4.7,  +0.1,  +9.2,  TOL1)),
    ]
    for label, a, b, anchor in PAIRS:
        shared, d = deltas(per[a], per[b])
        row = add("vit", label, "D.7",
                  {s: per[a][s] - per[b][s] for s in shared},
                  "step≥2500 median", d, anchor)
        if label == "linSAM−SAM":
            # CI90 stays the TOST CI (90%, +-3.8 margin context) -- print it
            # explicitly so the TOST layer can be transcribed unchanged.
            row["note"] = (f"TOST CI90 [{row['ci90_lo']:+.2f},"
                           f"{row['ci90_hi']:+.2f}] (keep 90%)")


# ------------------------------------------------------ 6. D.8 200ep
# analyze_long_recipe.py L41 (EPOCH=391), L54-63 (200ep recipe: steps
# 78200, steady epochs 30-120), L105-110 (complete-run filter: last step
# >= STEPS-EPOCH), L116-117 (steady-window mean lambda_max), L152-158
# (paired vs SGD on shared seeds).
def g6_200ep():
    EPOCH, STEPS = 391, 78200
    lo, hi = 30 * EPOCH, 120 * EPOCH
    per = {name: window_stat(p, "lambda_max", lo, hi, hi_inc=True,
                             min_last_step=STEPS - EPOCH)
           for name, p in {
               "SGD":    "cifar100_r18_sgd_200ep_s",
               "rank-1": "cifar100_r18_rankkx_d1_200ep_s",
               "SAM":    "cifar100_r18_sam_200ep_s"}.items()}
    for name, anchor in (("SAM", (-5.58, -6.25, -4.91, TOL2)),
                         ("rank-1", (+1.60, -1.89, +5.09, TOL2))):
        shared, d = deltas(per[name], per["SGD"])
        add("200ep", f"Δ({name}−SGD)", "D.8",
            {s: per[name][s] - per["SGD"][s] for s in shared},
            "epochs 30–120 mean", d, anchor)


# -------------------------------------------------- 7. D.11 rho sweep
# analyze_rho_sweep.py L28 (LATE_STEP=2000), L32-40 (ARM_RHOS; rho=0.05
# comes from the mb reruns), L40 (SGD_PREFIX), L61-72 (loader), L109-112
# (paired vs SGD on shared seeds).
def g7_rho():
    RHO_NEW = [0.005, 0.01, 0.02, 0.1]
    ARM_RHOS = {
        "sam":    {**{r: f"cifar100_r18_sam_rho{r}_s" for r in RHO_NEW},
                   0.05: "cifar100_r18_sam_mb_s"},
        "linsam": {**{r: f"cifar100_r18_linsam_rho{r}_s" for r in RHO_NEW},
                   0.05: "cifar100_r18_linsam_mb_s"},
    }
    sgd = window_stat("cifar100_r18_sgd_detarpack_s", "lambda_max", 2000)
    for arm, rhos in ARM_RHOS.items():
        for r, pfx in sorted(rhos.items()):
            per = window_stat(pfx, "lambda_max", 2000)
            if not per:
                continue
            shared, d = deltas(per, sgd)
            add("rho", f"Δ({arm}(ρ={r})−SGD)", "D.11",
                {s: per[s] - sgd[s] for s in shared},
                f"step≥2000 mean (n_ρ={len(per)})", d)
    # anchor: reproduce the existing figures/rho_sweep_summary.csv CI90s
    # (same generator conventions) -- a self-consistency gate for the group.
    csv = ROOT / "figures" / "rho_sweep_summary.csv"
    if not csv.exists():
        return
    ref = pd.read_csv(csv)
    ok, tot = 0, 0
    for _, rr in ref.iterrows():
        mine = [r for r in ROWS if r["group"] == "rho"
                and r["comparison"] ==
                f"Δ({rr.arm}(ρ={rr.rho:g})−SGD)"]
        if not mine or not (rr.paired_ci90_lo == rr.paired_ci90_lo):
            continue
        tot += 1
        m = mine[0]
        ok += (abs(m["ci90_lo"] - rr.paired_ci90_lo) <= 0.005
               and abs(m["ci90_hi"] - rr.paired_ci90_hi) <= 0.005
               and abs(m["mean_delta"] - rr.paired_delta_sgd) <= 0.005)
    for r in ROWS:
        if r["group"] == "rho":
            r["anchor"] = "existing rho_sweep_summary.csv CI90"
            r["anchor_ok"] = str(ok == tot and tot > 0)
            r["note"] = f"{ok}/{tot} CSV rows matched"


# --------------------------------------- 8. D.11 tau interpolation
# analyze_tau_interpolation.py L56 (LATE_STEP=2000), L59 (BSHARP_COL =
# lhg_bs_mean -- the paper's paired lstep contrast), L62-67 (TAU_ARMS),
# L107/L112-113 (window means).
def g8_tau():
    TAU = [(0.00, "cifar100_r18_linsam_bnmatched_mb_s"),
           (0.25, "cifar100_r18_sam_interp_t0.25_mb_s"),
           (0.50, "cifar100_r18_sam_interp_t0.5_mb_s"),
           (1.00, "cifar100_r18_sam_mb_s")]
    per = {tau: window_stat(p, "lhg_bs_mean", 2000) for tau, p in TAU}
    for tau, _ in TAU[1:]:
        shared, d = deltas(per[tau], per[0.00])
        anchor = (-3.67, -5.42, -1.92, TOL2) if tau == 1.00 else None
        add("tau", f"λstep Δ(τ={tau:.2f}−τ=0.00)",
            "D.11" if tau == 1.00 else "extra",
            {s: per[tau][s] - per[0.00][s] for s in shared},
            "step≥2000 mean (lhg_bs_mean)", d, anchor)


# ------------------------------------------ 9. D.11 negative-sign arm
# analyze_linsamneg.py L41 (LATE_STEP), L43-47 (prefixes: SGD lambda_max
# baseline = sgd_detarpack; linSAM = linsam_mb; bsharp baseline =
# sgd_detarpack_mb2), L96-97 (late mean), L125-149/161-169 (paired CIs).
def g9_linsamneg():
    NEG = "cifar100_r18_linsamneg_mb_s"
    specs = [
        ("Δ(linsamneg−SGD) λmax", NEG, "cifar100_r18_sgd_detarpack_s",
         "lambda_max", (+2.07, -0.71, +4.86, TOL2)),
        ("Δ(linsamneg−linSAM) λmax", NEG, "cifar100_r18_linsam_mb_s",
         "lambda_max", (-3.20, -5.32, -1.08, TOL2)),
        ("Δ(linsamneg−SGDmb2) bsharp", NEG, "cifar100_r18_sgd_detarpack_mb2_s",
         "lhg_bs_mean", (+0.43, -1.72, +2.58, TOL2)),
    ]
    for label, pa, pb, col, anchor in specs:
        A = window_stat(pa, col, 2000)
        B = window_stat(pb, col, 2000)
        shared, d = deltas(A, B)
        add("linsamneg", label, "D.11",
            {s: A[s] - B[s] for s in shared}, "step≥2000 mean", d, anchor)


# ------------------------------------------------------------------ main
def main():
    for g in (g1_bsharp, g2_rankk, g3_regime, g4_bn, g5_vit, g6_200ep,
              g7_rho, g8_tau, g9_linsamneg):
        g()

    df = pd.DataFrame(ROWS)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    # ---- 1) anchor gate table ------------------------------------------------
    anch = df[df.anchor != ""]
    n_pass = int((anch.anchor_ok == "True").sum())
    print(f"## ROUND 27 anchor gate — {n_pass}/{len(anch)} passed\n")
    print("| # | group | comparison | paper anchor | recomputed CI90 | verdict |")
    print("|---|-------|------------|--------------|-----------------|---------|")
    for i, r in enumerate(anch.itertuples(), 1):
        ok = r.anchor_ok == "True"
        print(f"| {i} | {r.group} | {r.comparison} | {r.anchor} | "
              f"{r.mean_delta:+.2f} [{r.ci90_lo:+.2f},{r.ci90_hi:+.2f}] | "
              f"{'PASS' if ok else '**FAILED-ANCHOR**'} |")
        if not ok:
            print(f"|   |       |            | per-seed Δ: {r.per_seed_deltas} | | |")

    # ---- 2) full CI95 table --------------------------------------------------
    print(f"\n## ROUND 27 CI95 table (ordinary paired differences, "
          f"t.ppf(0.975, n−1))\n")
    print("| comparison | n | window | meanΔ | CI90 (anchor) | CI95 | resolved95 |")
    print("|------------|---|--------|-------|---------------|------|------------|")
    for r in df.itertuples():
        res = {"True": "yes", "False": "no", "": "n<2"}[r.resolved95]
        print(f"| {r.comparison} | {r.n} | {r.window} | {r.mean_delta:+.2f} | "
              f"[{r.ci90_lo:+.2f},{r.ci90_hi:+.2f}] | "
              f"[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}] | {res} |")

    # ---- 3) rho first-resolved summary (expectation check, not a target) ------
    rho_rows = [r for r in ROWS if r["group"] == "rho"]
    for arm in ("sam", "linsam"):
        first = next((r for r in rho_rows
                      if r["comparison"].startswith(f"Δ({arm}(")
                      and r["resolved95"] == "True"), None)
        if first:
            print(f"\nrho sweep: first resolved95 for {arm}: "
                  f"{first['comparison']} {first['mean_delta']:+.2f} "
                  f"[{first['ci95_lo']:+.2f},{first['ci95_hi']:+.2f}]")
        else:
            print(f"\nrho sweep: no resolved95 for {arm} at any tested rho")

    print(f"\nwrote {OUT_CSV} ({len(df)} rows); "
          f"anchor gate {n_pass}/{len(anch)}")


if __name__ == "__main__":
    main()
