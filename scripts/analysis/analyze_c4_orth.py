"""Analyze C4 orthogonal-damping sweep (d ∈ {1, 1.5, 2, 4, 8} × 5 seeds).

Reads:  runs/c4_orth_d{D}_s{0..4}/manifest.json + per_step.csv + metrics.csv
Writes: runs/c4/ANALYSIS.md + runs/c4/analysis.json
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("runs")
OUT = Path("runs/c4")
OUT.mkdir(parents=True, exist_ok=True)

DS = [1, 1.5, 2, 4, 8]
SEEDS = [0, 1, 2, 3, 4]
TWO_OVER_ETA = 40.0


def ci95(xs):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    n = xs.size
    if n < 2: return 0.0
    tcrit = {1: float("inf"), 2: 4.303, 3: 3.182, 4: 2.776}.get(n-1, 1.96)
    return float(tcrit * xs.std(ddof=1) / math.sqrt(n))


def load_d(d):
    """Load all 5 seeds for one d-value. Handles both int and float dir names
    (d=1 → 'c4_orth_d1.0_s0', d=1.5 → 'c4_orth_d1.5_s0')."""
    rows = []
    for s in SEEDS:
        # try float-formatted name (what the runner produces: float(D_VALUE))
        for name in [f"c4_orth_d{float(d)}_s{s}", f"c4_orth_d{d}_s{s}"]:
            mf = RUNS / name / "manifest.json"
            if mf.exists():
                rows.append(json.loads(mf.read_text()))
                break
    return rows


def main():
    report = {"d_values": DS, "seeds": SEEDS, "two_over_eta": TWO_OVER_ETA}
    table = []

    for d in DS:
        runs = load_d(d)
        if not runs: continue
        lams = [r["lambda_max_final"] for r in runs]
        losses = [r["loss_final"] for r in runs]
        xts = [r["xt_step500_mean"] for r in runs]
        oscs = [r["xt_oscillation_window50"] for r in runs]
        divs = [r["diverged"] for r in runs]
        peaks = [r["lambda_max_peak"] for r in runs]
        entry = {
            "d": d, "n_seeds": len(runs),
            "lambda_max_final_mean": float(np.mean(lams)),
            "lambda_max_final_ci": ci95(lams),
            "lambda_max_final_per_seed": lams,
            "lambda_max_peak_mean": float(np.mean(peaks)),
            "loss_final_mean": float(np.mean(losses)),
            "loss_final_ci": ci95(losses),
            "xt_mean": float(np.mean(xts)),
            "xt_ci": ci95(xts),
            "oscillation_mean": float(np.mean(oscs)),
            "diverged_count": int(sum(divs)),
            "diverged_frac": float(np.mean(divs)),
        }
        table.append(entry)
    report["table"] = table

    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, default=str))

    # markdown
    lines = ["# C4 Orthogonal Damping — Analysis", "",
             f"Setup: tanh MLP [20-256-256-1], teacher-student full-batch GD, "
             f"η=0.05, 2/η={TWO_OVER_ETA:g}, 3000 steps, 5 seeds. "
             f"Δθ = −η(g_∥ + (1/d)g_⊥). d=1 ⇒ vanilla SGD.", "",
             "## Summary table (5-seed mean ± 95% CI)", "",
             "| d | λ_max final (mean±CI) | λ_max peak | loss final | \\|x_t\\| mean | oscillation | diverged |",
             "|---|---|---|---|---|---|---|"]
    for e in table:
        lines.append(f"| {e['d']} | {e['lambda_max_final_mean']:.2f} ± {e['lambda_max_final_ci']:.2f} | "
                     f"{e['lambda_max_peak_mean']:.1f} | {e['loss_final_mean']:.4f} ± {e['loss_final_ci']:.4f} | "
                     f"{e['xt_mean']:.4f} ± {e['xt_ci']:.4f} | {e['oscillation_mean']:.4f} | "
                     f"{e['diverged_count']}/{e['n_seeds']} |")
    lines.append("")

    # sanity check
    d1 = next((e for e in table if e["d"] == 1), None)
    if d1:
        pin = abs(d1["lambda_max_final_mean"] - TWO_OVER_ETA) < 5.0
        lines.append(f"## Sanity: d=1 ≈ vanilla SGD")
        lines.append(f"- λ_max final = {d1['lambda_max_final_mean']:.2f} ± {d1['lambda_max_final_ci']:.2f} "
                     f"(expected ~{TWO_OVER_ETA:g}): **{'PASS' if pin else 'FAIL'}**")
        lines.append("")

    # key finding
    d8 = next((e for e in table if e["d"] == 8), None)
    d_large = [e for e in table if e["d"] >= 4]
    lines.append("## Key finding")
    if d8:
        diverged_8 = d8["diverged_count"] > 0
        pin_8 = abs(d8["lambda_max_final_mean"] - TWO_OVER_ETA) < 10.0
        if diverged_8:
            lines.append(f"- d=8: **DIVERGED** in {d8['diverged_count']}/{d8['n_seeds']} seeds — "
                         f"strong orthogonal damping is unstable.")
        elif pin_8:
            lines.append(f"- d=8: λ_max = {d8['lambda_max_final_mean']:.2f} ± {d8['lambda_max_final_ci']:.2f} "
                         f"— **CAP ROBUST**: the EoS self-stabilization survives even strong orthogonal damping. "
                         f"λ_max stays pinned at 2/η regardless of d.")
        else:
            lines.append(f"- d=8: λ_max = {d8['lambda_max_final_mean']:.2f} — neither clean pin nor divergence. "
                         f"Intermediate regime.")
    lines.append("")

    text = "\n".join(lines)
    (OUT / "ANALYSIS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
