"""Round-7 BN-protocol controls — normalization-confound analysis.

The eval-mode reference λ_max is measured THROUGH the BN running statistics,
and sam_step evolves them twice per step where sgd/linsam evolve them once.
The three r7 controls (bit-verified against sam_step's buffer evolution by
scripts/analysis/validate_bn_match.py) isolate that confound:

  sam_bnfreeze      : training = sam, running stats FROZEN
  sgd_bn2x          : update = sgd, BN evolves exactly as in sam
  linsam_bnmatched  : update = first-order linSAM, BN evolves as in sam

This script reports, per arm (r7 three + the sam_mb / linsam_mb baselines):
  * late-window reference λ_max mean ± 95% CI (t-based, steps ≥ 2000),
  * paired Δ vs the SGD baseline on common seeds (90% CI),
  * the BN-protocol contrast table:
        sam           vs sam_bnfreeze      (does freezing stats matter?)
        sgd           vs sgd_bn2x          (does the extra BN update alone
                                            move the measured λ_max?)
        linsam        vs linsam_bnmatched  (same question at fixed gradient)
    each as paired Δ with 90% CI + TOST equivalence test at the
    preregistered margin δ = 0.25·|late(sgd) − late(sam)| (Finding-41
    calibration: "equivalent" is used ONLY where TOST passes; otherwise
    indistinguishable / no measurable reduction).

SGD baseline family selection: the r7 arms carry the FULL mb measurement
schema (incl. lhg_bs_*), recorded identically in
cifar100_r18_sgd_detarpack_mb2_s* — chosen when its manifest flags match the
r7 arms exactly (bs/steps/momentum/lr/multibatch/outcome_*); the legacy
cifar100_r18_sgd_detarpack_mb_s* (round-2, no lhg_bs columns) is a fallback.

Missing seeds are skipped gracefully (partial n analyses are first-class).

Smoke test (n=1, 100-step r7 smokes):
    .venv/bin/python scripts/analysis/analyze_r7_bn.py --late-step 0 \
        --arm sam_bnfreeze=smoke_r7b_sam_bnfreeze_s \
        --arm sgd_bn2x=smoke_r7b_sgd_bn2x_s \
        --arm linsam_bnmatched=smoke_r7b_linsam_bnmatched_s

Outputs: figures/r7_bn_summary.csv (+ stdout tables).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LATE_STEP = 2000
FLAGS = ["bs", "steps", "momentum", "lr", "multibatch", "outcome_arpack",
         "outcome_every", "eval_batch", "schedule"]

R7_ARMS = {          # arm -> default run prefix
    "sam_bnfreeze": "cifar100_r18_sam_bnfreeze_mb_s",
    "sgd_bn2x": "cifar100_r18_sgd_bn2x_mb_s",
    "linsam_bnmatched": "cifar100_r18_linsam_bnmatched_mb_s",
}
BASE_ARMS = {
    "sam": "cifar100_r18_sam_mb_s",
    "linsam": "cifar100_r18_linsam_mb_s",
}
SGD_CANDIDATES = ["cifar100_r18_sgd_detarpack_mb2_s",  # preferred (lhg_bs cols)
                  "cifar100_r18_sgd_detarpack_mb_s"]   # legacy fallback


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_ci(d, level=0.90):
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean())
    if n < 2:
        return m, 0.0, m, m
    h = float(stats.t.ppf(1 - (1 - level) / 2, n - 1) * d.std(ddof=1) / np.sqrt(n))
    return m, h, m - h, m + h


def load(prefix):
    out = {}
    for d in sorted(Path("runs").glob(prefix + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            out[int(s)] = pd.read_csv(d / "metrics.csv")
    return out


def late_mean(df, late_step):
    m = df[df["step"] >= late_step]
    if m.empty:
        return float("nan")
    return float(m["lambda_max"].mean())


def manifest_of(prefix, seed):
    p = Path("runs") / f"{prefix}{seed}" / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def flags_match(ma, mb):
    if ma is None or mb is None:
        return False
    return all(ma.get(k) == mb.get(k) for k in FLAGS)


def tost(d, delta, alpha=0.05):
    """Two one-sided t-tests on paired diffs: reject H0: |mean| ≥ δ."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or delta <= 0:
        return dict(n=n, delta=delta, reject=False, why="n<2 or δ≤0")
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        # zero-variance: equivalent iff |mean| < δ strictly
        return dict(n=n, delta=delta, reject=abs(float(d.mean())) < delta,
                    why="zero-variance")
    tcrit = float(stats.t.ppf(1 - alpha, n - 1))
    t_low = (float(d.mean()) + delta) / se    # H01: mean ≤ −δ
    t_high = (float(d.mean()) - delta) / se   # H02: mean ≥ +δ  (negated)
    return dict(n=n, delta=delta, reject=bool(t_low > tcrit and -t_high > tcrit),
                why=f"t_low={t_low:.2f}, t_high={-t_high:.2f}, crit={tcrit:.2f}")


def wording(m, lo, hi, tost_res):
    """Finding-41 calibrated phrasing: 'equivalent' ONLY on TOST pass."""
    if tost_res["reject"]:
        return f"equivalent (TOST δ={tost_res['delta']:.2f})"
    if lo <= 0.0 <= hi:
        return "indistinguishable (90% CI covers 0)"
    return ("no measurable reduction" if m >= 0 else "measurable reduction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--late-step", type=int, default=LATE_STEP)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PREFIX",
                    help="override an arm's run prefix (repeatable); e.g. "
                         "sam_bnfreeze=smoke_r7b_sam_bnfreeze_s")
    args = ap.parse_args()
    LATE = args.late_step

    arms = dict(R7_ARMS)
    arms.update(BASE_ARMS)
    for ov in args.arm:
        k, v = ov.split("=", 1)
        arms[k] = v

    # ---- SGD baseline family: exact-flag match against the r7 protocol ----
    r7_ref_manifest = (manifest_of(R7_ARMS["sam_bnfreeze"], 0)
                       or manifest_of(BASE_ARMS["sam"], 0))
    sgd_prefix = None
    for cand in SGD_CANDIDATES:
        m = manifest_of(cand, 0)
        if m and flags_match(m, r7_ref_manifest):
            sgd_prefix = cand
            print(f"SGD baseline family: {cand} (manifest flags match r7 "
                  f"protocol on {len(FLAGS)} fields)")
            break
    if sgd_prefix is None:
        sgd_prefix = SGD_CANDIDATES[0]
        print(f"[warn] no exact-flag SGD candidate matched; defaulting to "
              f"{sgd_prefix}")
    arms["sgd"] = sgd_prefix

    runs = {name: load(pfx) for name, pfx in arms.items()}
    for name, r in runs.items():
        print(f"{name:18s} seeds: {sorted(r) if r else 'NONE (skipped)'}")
    late = {name: {s: late_mean(df, LATE) for s, df in r.items()}
            for name, r in runs.items()}

    rows = []
    print(f"\n--- late-window reference λmax (steps ≥ {LATE}) ---")
    for name in ("sgd", "sam", "linsam", "sam_bnfreeze", "sgd_bn2x",
                 "linsam_bnmatched"):
        per = [v for v in late.get(name, {}).values() if not np.isnan(v)]
        if not per:
            print(f"{name:18s} no completed runs yet — skipped")
            continue
        m, h = ci95(per)
        rows.append(dict(arm=name, quantity="late_lmax", value=m, ci95=h,
                         n=len(per)))
        print(f"{name:18s} {m:7.2f} ± {h:5.2f}  (n={len(per)})")

    sgd_late = late.get("sgd", {})
    print("\n--- paired Δ vs SGD baseline (control − sgd, common seeds) ---")
    for name in ("sam", "linsam", "sam_bnfreeze", "sgd_bn2x",
                 "linsam_bnmatched"):
        common = sorted(set(late.get(name, {})) & set(sgd_late))
        if not common:
            print(f"{name:18s} no common seeds with sgd — skipped")
            continue
        d = [late[name][s] - sgd_late[s] for s in common]
        m, h, lo, hi = paired_ci(d)
        rows.append(dict(arm=f"Δ({name}−sgd)", quantity="paired_lmax_delta",
                         value=m, ci90=h, lo=lo, hi=hi, n=len(d)))
        print(f"Δ({name:14s}−sgd) = {m:+8.2f}  90% CI [{lo:+8.2f},{hi:+8.2f}]"
              f"  seeds {common}")

    # ---- BN-protocol contrast table + TOST ----
    print("\n--- BN-protocol contrast table "
          "(TOST margin δ = 0.25·|late(sgd) − late(sam)|) ---")
    mu_sgd = np.nanmean([v for v in sgd_late.values() if not np.isnan(v)]
                        ) if sgd_late else float("nan")
    mu_sam = np.nanmean([v for v in late.get("sam", {}).values()
                         if not np.isnan(v)]) if late.get("sam") else float("nan")
    delta = 0.25 * abs(mu_sgd - mu_sam) if not (np.isnan(mu_sgd) or
                                                np.isnan(mu_sam)) else float("nan")
    print(f"margin δ = 0.25·|{mu_sgd:.2f} − {mu_sam:.2f}| = {delta:.2f}")
    CONTRASTS = [("sam", "sam_bnfreeze", "freeze running stats"),
                 ("sgd", "sgd_bn2x", "extra BN update alone"),
                 ("linsam", "linsam_bnmatched", "extra BN update, 1st-order grad")]
    for ref, ctrl, label in CONTRASTS:
        common = sorted(set(late.get(ref, {})) & set(late.get(ctrl, {})))
        if not common:
            print(f"{ctrl:18s} vs {ref:8s} — no common seeds — skipped")
            continue
        d = [late[ctrl][s] - late[ref][s] for s in common]
        m, h, lo, hi = paired_ci(d)
        t = tost(d, delta)
        w = wording(m, lo, hi, t)
        rows.append(dict(arm=f"{ctrl} vs {ref}", quantity="bn_contrast",
                         value=m, ci90=h, lo=lo, hi=hi, n=len(d),
                         tost_delta=t["delta"], tost_pass=t["reject"],
                         verdict=w))
        print(f"{ctrl:18s} − {ref:8s} = {m:+7.2f}  90% CI [{lo:+7.2f},{hi:+7.2f}]"
              f"  TOST {'PASS' if t['reject'] else 'fail'}  → {w}   [{label}]")

    out = Path("figures/r7_bn_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n=> {out}")


if __name__ == "__main__":
    main()
