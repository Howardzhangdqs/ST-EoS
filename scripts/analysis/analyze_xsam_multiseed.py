"""Analyze the X-SAM multi-seed experiment (Reviewer #2).

Addresses two of reviewer #2's weaknesses:
  (1) n=1 single-seed → n=5 here (originally n=3; s3, s4 added after the
      reviewer noted the paired Δλ_max CI ±4.87 was too wide for the
      "did-not-replicate" claim).
  (2) measured λ̂_g (lower-bound proxy) not true λ_max → ARPACK λ_max here,
      alongside λ̂_g for the proxy-vs-true contrast.

The central question (Duan et al. 2026 λ_max claim replication): does the
faithful X-SAM rank-1 correction (Eq. 11, fixed α=0.2, sign-gated,
unconditional) drive λ_max BELOW plain SAM? With n=5 seeds we compute a
Δλ_max(faithful−sam) ± CI and check whether it crosses zero; we ALSO report
the **effect-size upper bound** (the high side of the CI), so the wording can
be "no significant effect detected at this power; effect-size upper bound Y".

Reads:
    runs/cifar10_r18_{sgd,sam,xsam_faithful,xsam_clamped}_s{0..4}/metrics.csv
    (must have the ARPACK column lambda_max + lambda_hat_grad + lambda_max_arpack_ritz)
    Auto-discovers available seeds — produces n=N analysis for whatever is present.

Writes:
    runs/xsam/ANALYSIS.md
    runs/xsam/analysis.json
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

RUN_ROOT = Path("runs")
OUT_DIR = Path("runs/xsam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPTS = ["sgd", "sam", "xsam_faithful", "xsam_clamped"]
# Auto-discover available seeds — handles n=3 → n=5 → n=10 transitions.
ALL_SEEDS = list(range(10))  # [0, 1, 2, ..., 9]

# Phase boundaries (in step count) — early/mid/late. The X-SAM runs go to 5000
# steps. We slice: early [0,1000), mid [1000,3000), late [3000,5000].
PHASES = [("early", 0, 1000), ("mid", 1000, 3000), ("late", 3000, 5000)]


def discover_seeds():
    """Return the list of seeds for which ALL 4 opts have ARPACK data."""
    present = []
    for s in ALL_SEEDS:
        if all((RUN_ROOT / f"cifar10_r18_{opt}_s{s}/metrics.csv").exists()
               for opt in OPTS):
            # also verify ARPACK column presence
            ok = True
            for opt in OPTS:
                df = pd.read_csv(RUN_ROOT / f"cifar10_r18_{opt}_s{s}/metrics.csv")
                if "lambda_max_arpack_ritz" not in df.columns:
                    ok = False
                    break
            if ok:
                present.append(s)
    return present


SEEDS = discover_seeds()


def load_run(opt, seed):
    d = RUN_ROOT / f"cifar10_r18_{opt}_s{seed}"
    mpath = d / "metrics.csv"
    if not mpath.exists():
        return None
    df = pd.read_csv(mpath)
    # the ARPACK-measured runs have lambda_max_arpack_ritz column
    if "lambda_max_arpack_ritz" not in df.columns:
        return None
    mpath2 = d / "manifest.json"
    manifest = json.loads(mpath2.read_text()) if mpath2.exists() else {}
    return {"df": df, "manifest": manifest, "dir": d}


def t_ci95(mean, sd, n):
    """Two-sided 95% CI half-width using Student t (df=n-1).
    n=3 → 4.303, n=4 → 3.182, n=5 → 2.776 (monotonically tighter as n grows;
    the n=3 → n=5 move tightens CI by ~35%)."""
    if n < 2:
        return float("inf")
    t_crit = {1: float("inf"), 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    return t_crit * sd / math.sqrt(n)


def phase_stats_one_run(df, col):
    """Per-phase mean / median of a column. Skip step 0 (random-init transient
    — λ_max huge there, not representative of any phase's 'shape')."""
    out = {}
    for name, lo, hi in PHASES:
        sub = df[(df["step"] > 0) & (df["step"] >= lo) & (df["step"] < hi)][col]
        sub = sub.dropna()
        if len(sub) == 0:
            out[name] = {"mean": float("nan"), "median": float("nan"), "n": 0}
        else:
            out[name] = {"mean": float(sub.mean()), "median": float(sub.median()),
                         "n": int(len(sub))}
    return out


def aggregate_across_seeds(runs, col, phase):
    """For one opt, one phase, across seeds: take per-seed phase-mean, then
    compute n=3 mean ± sd ± t-CI half-width."""
    seed_means = []
    for r in runs:
        if r is None:
            continue
        p = phase_stats_one_run(r["df"], col)[phase]
        if p["n"] > 0 and not np.isnan(p["mean"]):
            seed_means.append(p["mean"])
    if not seed_means:
        return None
    n = len(seed_means)
    mean = st.mean(seed_means)
    sd = st.stdev(seed_means) if n >= 2 else 0.0
    return {
        "n_seeds": n,
        "seed_means": seed_means,
        "mean": mean,
        "sd": sd,
        "min": min(seed_means),
        "max": max(seed_means),
        "ci95_half": t_ci95(mean, sd, n),
    }


def build_table(col):
    """n=3 mean ± CI table: opt × phase."""
    table = {}
    for opt in OPTS:
        runs = [load_run(opt, s) for s in SEEDS]
        table[opt] = {}
        for phase_name, _, _ in PHASES:
            table[opt][phase_name] = aggregate_across_seeds(runs, col, phase_name)
    return table


def delta_faithful_minus_sam(table):
    """Per-phase Δ = faithful − sam, with propagation of CI.
    Δ mean = mean_f − mean_s. Δ CI half-width = sqrt(ci_f² + ci_s²) (Gaussian
    propagation; conservative for small n but standard practice)."""
    out = {}
    for phase_name, _, _ in PHASES:
        f = table["xsam_faithful"][phase_name]
        s = table["sam"][phase_name]
        if f is None or s is None:
            out[phase_name] = None
            continue
        delta_mean = f["mean"] - s["mean"]
        # also compute per-seed paired delta if seeds align
        paired = []
        for sf, ss in zip(f["seed_means"], s["seed_means"]):
            paired.append(sf - ss)
        if len(paired) >= 2:
            pmean = st.mean(paired)
            psd = st.stdev(paired)
            pci_half = t_ci95(pmean, psd, len(paired))
        else:
            pmean, pci_half = float("nan"), float("inf")
        out[phase_name] = {
            "delta_mean": delta_mean,
            "delta_ci95_half_gauss": math.sqrt(f["ci95_half"]**2 + s["ci95_half"]**2),
            "paired_delta_per_seed": paired,
            "paired_delta_mean": pmean,
            "paired_delta_ci95_half": pci_half,
            "crosses_zero_paired": (pmean - pci_half <= 0 <= pmean + pci_half),
        }
    return out


def classify_delta(delta):
    """Decision per preregistration-style判据:
      CI crosses 0 → 'did not replicate Duan' (降级措辞)
      CI entirely < 0 → 'replicated Duan' (faithful reduces λ_max below SAM)
      CI entirely > 0 → 'reverse direction' (faithful increases λ_max)
    Uses the paired CI (more honest for same-seed comparison)."""
    if delta is None:
        return "inconclusive"
    lo = delta["paired_delta_mean"] - delta["paired_delta_ci95_half"]
    hi = delta["paired_delta_mean"] + delta["paired_delta_ci95_half"]
    if lo <= 0 <= hi:
        return "did_not_replicate (CI crosses 0)"
    if hi < 0:
        return "replicated (faithful λ_max significantly < SAM)"
    return "reverse (faithful λ_max significantly > SAM)"


def effect_size_upper_bound(delta):
    """The 'effect-size upper bound' Y: the largest effect consistent with the
    data at the present power. This is the HIGH side of the paired CI (a one-
    sided 97.5% bound, conservative). Use for the power-aware wording:
      'We did not detect a statistically significant effect at this power;
       the effect-size upper bound is Y.'
    For Δ(faithful − sam), negative = Duan replicated (faithful reduces λ_max
    below SAM), so the upper bound is the LEAST favorable to Duan still in the
    CI: hi = mean + ci_half."""
    if delta is None:
        return float("nan")
    return delta["paired_delta_mean"] + delta["paired_delta_ci95_half"]


def fmt_ci(stat):
    if stat is None:
        return "—"
    return (f"{stat['mean']:6.2f} ± {stat['ci95_half']:5.2f}  "
            f"(range {stat['min']:.1f}–{stat['max']:.1f}, n={stat['n_seeds']})")


def main():
    n = len(SEEDS)
    dfree = n - 1
    report = {"opts": OPTS, "seeds": SEEDS, "n": n,
              "phases": [p[0] for p in PHASES]}

    lam_table = build_table("lambda_max")
    lhg_table = build_table("lambda_hat_grad")
    delta_lam = delta_faithful_minus_sam(lam_table)
    delta_lhg = delta_faithful_minus_sam(lhg_table)

    report["lambda_max_arpack_table"] = lam_table
    report["lambda_hat_grad_table"] = lhg_table
    report["delta_faithful_minus_sam_lambda_max"] = delta_lam
    report["delta_faithful_minus_sam_lambda_hat_grad"] = delta_lhg
    report["classification"] = {
        phase: classify_delta(delta_lam[phase]) for phase in [p[0] for p in PHASES]
    }
    report["effect_size_upper_bound_late_lambda_max"] = (
        effect_size_upper_bound(delta_lam["late"]) if delta_lam.get("late") else float("nan"))

    # final acc / loss table
    final = {}
    for opt in OPTS:
        final[opt] = {}
        accs, losses = [], []
        for s in SEEDS:
            r = load_run(opt, s)
            if r is None or len(r["df"]) == 0:
                continue
            accs.append(float(r["df"]["eval_acc"].iloc[-1]))
            losses.append(float(r["df"]["train_loss"].iloc[-1]))
        if accs:
            accs_arr = np.asarray(accs)
            losses_arr = np.asarray(losses)
            final[opt] = {
                "final_acc_mean": float(accs_arr.mean()),
                "final_acc_ci95": ci95(accs),
                "final_acc_range": [float(accs_arr.min()), float(accs_arr.max())],
                "final_loss_mean": float(losses_arr.mean()),
                "final_loss_ci95": ci95(losses),
                "n_seeds": len(accs),
            }
    report["final_acc_loss"] = final

    (OUT_DIR / "analysis.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- human briefing (writes to ANALYSIS_n{N}.md AND ANALYSIS.md) ----
    lines = [f"# X-SAM Multi-Seed Analysis (Reviewer #2) — n={n}", "",
             f"Setup: CIFAR-10/ResNet-18, lr=0.1, μ=0.9, wd=5e-4, bs=128, 5000 steps, "
             f"deterministic. ARPACK λ_max (k=3, tol=1e-6, B=512) every 100 steps + λ̂_g. "
             f"**n={n} seeds** ({SEEDS}). Student-t df={dfree} "
             f"(t-crit {TCRIT_LABEL[dfree]}).", ""]

    lines.append(f"## λ_max (ARPACK, true) — n={n} mean ± 95% CI (Student t, df={dfree})")
    lines.append("")
    lines.append("| opt | early (0–1k) | mid (1–3k) | late (3–5k) |")
    lines.append("|---|---|---|---|")
    for opt in OPTS:
        row = [f"`{opt}`"]
        for phase_name, _, _ in PHASES:
            row.append(fmt_ci(lam_table[opt][phase_name]).replace("|", "/"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"## λ̂_g (gradient Rayleigh proxy) — n={n}, same table for proxy-vs-true contrast")
    lines.append("")
    lines.append("| opt | early | mid | late |")
    lines.append("|---|---|---|---|")
    for opt in OPTS:
        row = [f"`{opt}`"]
        for phase_name, _, _ in PHASES:
            row.append(fmt_ci(lhg_table[opt][phase_name]).replace("|", "/"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"## Δλ_max(faithful − sam) — the Duan et al. replication test (n={n})")
    lines.append("")
    lines.append("Negative Δ ⇒ faithful reduces λ_max below SAM (Duan replicated). "
                 "Positive Δ ⇒ reverse. CI crossing 0 ⇒ not significant (did not replicate). "
                 "**Effect-size upper bound** = the high side of the paired CI "
                 "(the largest effect still consistent with the data at this power).")
    lines.append("")
    lines.append("| phase | paired Δ ± CI95 | effect-size upper bound | per-seed Δs | verdict |")
    lines.append("|---|---|---|---|---|")
    for phase_name, _, _ in PHASES:
        d = delta_lam[phase_name]
        if d is None:
            lines.append(f"| {phase_name} | — | — | — | inconclusive |")
            continue
        per_seed = ", ".join(f"{x:+.2f}" for x in d["paired_delta_per_seed"])
        verdict = report["classification"][phase_name]
        up_b = effect_size_upper_bound(d)
        lines.append(f"| {phase_name} | {d['paired_delta_mean']:+6.2f} ± "
                     f"{d['paired_delta_ci95_half']:5.2f} | {up_b:+.2f} "
                     f"| [{per_seed}] | {verdict} |")
    lines.append("")

    lines.append(f"## Δλ̂_g(faithful − sam) — proxy comparator (n={n}, same diffs on λ̂_g)")
    lines.append("")
    lines.append("| phase | paired Δ ± CI95 |")
    lines.append("|---|---|")
    for phase_name, _, _ in PHASES:
        d = delta_lhg[phase_name]
        if d is None:
            lines.append(f"| {phase_name} | — |")
            continue
        lines.append(f"| {phase_name} | {d['paired_delta_mean']:+6.2f} ± "
                     f"{d['paired_delta_ci95_half']:5.2f} |")
    lines.append("")

    lines.append(f"## Final accuracy / loss (n={n}, mean ± 95% CI)")
    lines.append("")
    lines.append("| opt | final acc (mean ± CI95) | final train loss (mean ± CI95) |")
    lines.append("|---|---|---|")
    for opt in OPTS:
        f = final.get(opt, {})
        if not f or f.get("n_seeds", 0) == 0:
            lines.append(f"| `{opt}` | — | — |")
            continue
        lines.append(f"| `{opt}` | {f['final_acc_mean']:.3f} ± {f['final_acc_ci95']:.3f} "
                     f"(range {f['final_acc_range'][0]:.3f}–{f['final_acc_range'][1]:.3f}) | "
                     f"{f['final_loss_mean']:.3f} ± {f['final_loss_ci95']:.3f} |")
    lines.append("")

    # Overall verdict + power-aware wording
    mid_verdict = report["classification"]["mid"]
    late_verdict = report["classification"]["late"]
    lines.append(f"## Overall Duan-replication verdict (n={n}, mid + late phases)")
    lines.append("")
    lines.append(f"- mid-phase: **{mid_verdict}**")
    lines.append(f"- late-phase: **{late_verdict}**")
    late_up = effect_size_upper_bound(delta_lam["late"]) if delta_lam.get("late") else float("nan")
    mid_up = effect_size_upper_bound(delta_lam["mid"]) if delta_lam.get("mid") else float("nan")
    lines.append(f"- late-phase effect-size upper bound: **{late_up:+.2f}**")
    lines.append(f"- mid-phase effect-size upper bound: **{mid_up:+.2f}**")
    lines.append("")

    if "did_not_replicate" in mid_verdict and "did_not_replicate" in late_verdict:
        lines.append(f"**Conclusion (n={n}):** Under our setting (CIFAR-10/ResNet-18, "
                     f"η=0.1, deterministic, n={n} seeds, ARPACK true λ_max), the faithful "
                     f"X-SAM rank-1 correction does NOT reduce λ_max below plain SAM "
                     f"by a statistically significant margin in either mid or late "
                     f"training. The Duan et al. λ_max gain is not replicated.")
        lines.append("")
        lines.append("**Recommended power-aware wording (for §5 / response):**")
        lines.append(f"> At n={n} deterministic seeds with ARPACK true-λ_max measurement, "
                     f"the paired Δλ_max(faithful − SAM) is "
                     f"{delta_lam['late']['paired_delta_mean']:+.2f} ± "
                     f"{delta_lam['late']['paired_delta_ci95_half']:.2f} at late training "
                     f"(CI crosses zero). We therefore detect no statistically significant "
                     f"effect of the rank-1 correction on λ_max beyond SAM's global "
                     f"smoothing at this power; **the late-phase effect-size upper bound "
                     f"is {late_up:+.2f}**, i.e. any reduction of λ_max below SAM is "
                     f"smaller than {abs(late_up):.1f} units in absolute value. The rank-1 "
                     f"correction's contribution to sharpness reduction is at most a "
                     f"{abs(late_up)/late_sam_baseline(lam_table):.0%} fraction of SAM's "
                     f"absolute late-phase λ_max, indistinguishable from zero at this power."
                     if not math.isnan(late_up) else "")
    elif "replicated" in mid_verdict or "replicated" in late_verdict:
        lines.append(f"**Conclusion (n={n}):** The Duan et al. λ_max gain replicates in at "
                     f"least one phase (faithful significantly < SAM). Report as "
                     f"replication.")
    else:
        lines.append(f"**Conclusion (n={n}):** Mixed / reverse direction. Report honestly.")

    text = "\n".join(lines)
    (OUT_DIR / f"ANALYSIS_n{n}.md").write_text(text)
    (OUT_DIR / "ANALYSIS.md").write_text(text)  # convenience: latest = canonical
    print(text)


def late_sam_baseline(lam_table):
    """Helper for the % effect-size: SAM's late-phase λ_max mean."""
    s = lam_table.get("sam", {}).get("late")
    return s["mean"] if s and not math.isnan(s["mean"]) else 1.0


# Critical-value label for human-readable output.
TCRIT_LABEL = {1: "∞", 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci95(x):
    """95% CI half-width (Student t)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    return t_ci95(float(np.mean(x)), sd, n)


if __name__ == "__main__":
    main()
