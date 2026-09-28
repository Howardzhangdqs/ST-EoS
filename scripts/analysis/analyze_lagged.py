"""Analyze lagged-phase plant sweep (Δθ-based + û-based arms).

Δθ-based arm: v̂ = unit(Δθ_{t−k}) — gradient/update direction, INACCURATE proxy.
û-based arm:  v̂ = û(t−k) — accurate power-iter eigenvector, lagged k steps.
              Isolates LAG from INACCURACY (the key discriminative control).

k ∈ {0, 1, 2, 3} × 5 seeds for dtheta; k ∈ {1, 2} × 5 seeds for eigvec.

Reads:  runs/lagged_k{K}_s{0..4}/manifest.json          (dtheta arm)
        runs/lagged_eig_k{K}_s{0..4}/manifest.json      (eigvec arm)
Writes: runs/lagged/ANALYSIS.md + runs/lagged/analysis.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("runs")
OUT = Path("runs/lagged")
OUT.mkdir(parents=True, exist_ok=True)

KS_DTHETA = [0, 1, 2, 3]
KS_EIGVEC = [1, 2, 3]
SEEDS = [0, 1, 2, 3, 4]
TWO_OVER_ETA = 40.0
ESCAPE_THRESH = 0.7 * TWO_OVER_ETA    # 28; escape if λ_max_final < 28
PIN_LO, PIN_HI = 0.7 * TWO_OVER_ETA, 1.5 * TWO_OVER_ETA  # [28, 60]


def ci95(xs):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    n = xs.size
    if n < 2: return 0.0
    tcrit = {1: float("inf"), 2: 4.303, 3: 3.182, 4: 2.776}.get(n-1, 1.96)
    return float(tcrit * xs.std(ddof=1) / math.sqrt(n))


def classify(lam_mean, diverged_frac):
    if diverged_frac > 0.5:
        return "DIVERGED"
    if lam_mean > 1.5 * TWO_OVER_ETA:
        return "SHARP-ESCAPE"
    if PIN_LO <= lam_mean <= PIN_HI:
        return "PIN at 2/η"
    return "INTERMEDIATE"


def load_arm(prefix, k):
    """Load all 5 seeds for one (prefix, k). prefix = 'lagged_k' or 'lagged_eig_k'."""
    rows = []
    for s in SEEDS:
        mf = RUNS / f"{prefix}{k}_s{s}" / "manifest.json"
        if not mf.exists(): continue
        rows.append(json.loads(mf.read_text()))
    return rows


def build_entry(k, runs, sensing):
    lams = [r["lambda_max_final"] for r in runs]
    losses = [r["loss_final"] for r in runs]
    xts = [r["xt_step500_mean"] for r in runs]
    oscs = [r["xt_oscillation_window50"] for r in runs]
    divs = [r["diverged"] for r in runs]
    peaks = [r["lambda_max_peak"] for r in runs]
    lam_mean = float(np.mean(lams))
    div_frac = float(np.mean(divs))
    return {
        "k": k, "sensing": sensing, "n_seeds": len(runs),
        "lambda_max_final_mean": lam_mean,
        "lambda_max_final_ci": ci95(lams),
        "lambda_max_final_per_seed": lams,
        "lambda_max_peak_mean": float(np.mean(peaks)),
        "loss_final_mean": float(np.mean(losses)),
        "loss_final_ci": ci95(losses),
        "xt_mean": float(np.mean(xts)),
        "xt_ci": ci95(xts),
        "oscillation_mean": float(np.mean(oscs)),
        "diverged_count": int(sum(divs)),
        "diverged_frac": div_frac,
        "classification": classify(lam_mean, div_frac),
    }


def main():
    report = {"two_over_eta": TWO_OVER_ETA}
    table = []
    # dtheta arm (existing)
    for k in KS_DTHETA:
        runs = load_arm("lagged_k", k)
        if runs:
            table.append(build_entry(k, runs, "dtheta"))
    # eigvec arm (new)
    for k in KS_EIGVEC:
        runs = load_arm("lagged_eig_k", k)
        if runs:
            table.append(build_entry(k, runs, "eigvec"))

    report["table"] = table
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, default=str))

    # split for display
    dtheta_rows = [e for e in table if e["sensing"] == "dtheta"]
    eigvec_rows = [e for e in table if e["sensing"] == "eigvec"]

    lines = ["# Lagged-Phase Plant — Analysis (Δθ-based + û-based arms)", "",
             f"Setup: tanh MLP [20-256-256-1], teacher-student full-batch GD, "
             f"η=0.05, 2/η={TWO_OVER_ETA:g}, 3000 steps, 5 seeds. "
             f"**Discriminative design:** Δθ-based arm confounds lag+inaccuracy; "
             f"û-based arm ISOLATES lag (accurate eigenvector, just stale).", "",
             "## Δθ-based arm (v̂ = unit(Δθ_{t−k}))", "",
             "| k | λ_max final (mean±CI) | λ_max peak | loss final | \\|x_t\\| mean | diverged | classification |",
             "|---|---|---|---|---|---|---|"]
    for e in dtheta_rows:
        lines.append(f"| {e['k']} | {e['lambda_max_final_mean']:.2f} ± {e['lambda_max_final_ci']:.2f} | "
                     f"{e['lambda_max_peak_mean']:.1f} | {e['loss_final_mean']:.4f} ± {e['loss_final_ci']:.4f} | "
                     f"{e['xt_mean']:.4f} | {e['diverged_count']}/{e['n_seeds']} | **{e['classification']}** |")
    lines.append("")

    lines.append("## û-based arm (v̂ = û(t−k), accurate power-iter eigenvector, lagged)")
    lines.append("")
    lines.append("| k | λ_max final (mean±CI) | λ_max peak | loss final | \\|x_t\\| mean | diverged | classification |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in eigvec_rows:
        lines.append(f"| {e['k']} | {e['lambda_max_final_mean']:.2f} ± {e['lambda_max_final_ci']:.2f} | "
                     f"{e['lambda_max_peak_mean']:.1f} | {e['loss_final_mean']:.4f} ± {e['loss_final_ci']:.4f} | "
                     f"{e['xt_mean']:.4f} | {e['diverged_count']}/{e['n_seeds']} | **{e['classification']}** |")
    lines.append("")

    # discriminative verdict
    lines.append("## Discriminative verdict (lag vs inaccuracy)")
    lines.append("")
    dt_k1 = next((e for e in dtheta_rows if e["k"] == 1), None)
    dt_k2 = next((e for e in dtheta_rows if e["k"] == 2), None)
    eg_k1 = next((e for e in eigvec_rows if e["k"] == 1), None)
    eg_k2 = next((e for e in eigvec_rows if e["k"] == 2), None)
    if eg_k1 and dt_k1:
        if "ESCAPE" in eg_k1["classification"]:
            lines.append(f"- û(t−1) **sharp-escapes** to {eg_k1['lambda_max_final_mean']:.0f} "
                         f"(same as k=0 accurate ≈ 262)")
            lines.append(f"- Δθ(t−1) **pins** at {dt_k1['lambda_max_final_mean']:.1f}")
            lines.append(f"- **CAUSAL ATTRIBUTION:** the Δθ-based pin is caused by **INACCURACY** "
                         f"(Δθ ≠ top eigenvector), NOT by lag. An accurate eigenvector û(t−k) "
                         f"with k-step lag still drives sharp-escape. "
                         f"**prop:lagged needs restructuring** — the operative cause is "
                         f"Δθ-direction being an inaccurate proxy for u₁, not staleness.")
        else:
            lines.append(f"- û(t−1) **pins** at {eg_k1['lambda_max_final_mean']:.1f}")
            lines.append(f"- **CAUSAL ATTRIBUTION:** lag itself causes the pin — even accurate "
                         f"eigenvectors at k-step lag re-impose the self-stabilization cap. "
                         f"prop:lagged holds.")
    lines.append("")

    text = "\n".join(lines)
    (OUT / "ANALYSIS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
