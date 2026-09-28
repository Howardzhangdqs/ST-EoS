"""Phase-2 diagnostic: validate λ_c ≈ λ_max and reproduce the EoS signature.

Reads runs/<name>/{metrics,sharpness}.csv and writes figures/eos.png with:
  (1) loss(t)              — long-run decrease + short-scale non-monotonicity
  (2) λ_max(t) & λ_c(t)    — both should climb then hug the 2/η threshold
  (3) ‖W_out‖(t)           — the Li et al. sharpness proxy

Also prints: corr(λ_max, λ_c), and how often |λ_max − 2/η| is within ±25%.
Usage: .venv/bin/python -m steos.analysis.plot_eos --run mlp_ts_eos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()

    run_dir = Path(args.run_root) / args.run
    m = pd.read_csv(run_dir / "metrics.csv")
    lr = float(pd.read_json(run_dir / "manifest.json", typ="series")["config"]["optimizer"]["lr"])
    two_over_eta = 2.0 / lr

    has_sharp = (run_dir / "sharpness.csv").exists()
    s = pd.read_csv(run_dir / "sharpness.csv") if has_sharp else None

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(m["step"], m["loss"], lw=0.8, color="tab:blue")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{args.run}: Edge-of-Stability diagnostic  (η={lr}, 2/η={two_over_eta:g})")
    axes[0].set_yscale("log")

    if s is not None:
        if "lambda_max" in s:
            axes[1].plot(s["step"], s["lambda_max"], "o-", ms=3, lw=0.8, label="λ_max (Hessian power iter)")
        if "lambda_c" in s:
            axes[1].plot(s["step"], s["lambda_c"], "s-", ms=3, lw=0.8, label="λ_c (critical sharpness)")
        axes[1].axhline(two_over_eta, color="red", ls="--", lw=1.2, label=f"2/η = {two_over_eta:g}")
        axes[1].set_ylabel("sharpness")
        axes[1].legend(loc="best", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "no sharpness.csv (set sharpness.enabled=true)", ha="center")

    axes[2].plot(m["step"], m["wnorm_output"], lw=0.8, color="tab:green", label="‖W_out‖  (sharpness proxy)")
    axes[2].plot(m["step"], m["update_norm"], lw=0.6, color="tab:orange", alpha=0.6, label="‖Δθ‖")
    axes[2].set_ylabel("norm")
    axes[2].set_xlabel("step")
    axes[2].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)

    out = run_dir / "figures" / "eos.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # ---- diagnostics ---------------------------------------------------------
    if s is not None and "lambda_max" in s and "lambda_c" in s:
        both = s.dropna(subset=["lambda_max", "lambda_c"])
        corr = both["lambda_max"].corr(both["lambda_c"])
        print(f"corr(λ_max, λ_c) = {corr:.4f}   (expect ≈1 on a locally-quadratic loss)")
        med_ratio = float((both["lambda_c"] / both["lambda_max"]).median())
        print(f"median(λ_c / λ_max) = {med_ratio:.3f}")
    if s is not None and "lambda_max" in s:
        lm = s["lambda_max"].dropna().to_numpy()
        near = float(np.mean(np.abs(lm - two_over_eta) <= 0.25 * two_over_eta))
        print(f"frac of steps with |λ_max − 2/η| ≤ 25%·(2/η): {near:.2f}  (EoS → high)")
        print(f"λ_max: init={lm[0]:.2f}  max={lm.max():.2f}  final={lm[-1]:.2f}")


if __name__ == "__main__":
    main()
