"""Sweep one config axis (default: learning rate) and collect a summary.

For each value it runs a full training (with curvature + snapshots), then
computes summary statistics: final/min loss, PCA explained variance of the
trajectory, λ_max init/max/final, the fraction of steps sitting at the EoS
threshold 2/η, and the final |cos(Δθ, v_top)| alignment.

Example:
    python scripts/experiments/sweep.py --config configs/mlp_synthetic_eos.yaml \
        --axis optimizer.lr --values 0.01,0.02,0.05,0.1,0.2 --steps 5000
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.analysis.trajectory import delta_vtop_alignment, load_run, pca_project
from steos.train import Trainer


def set_nested(d: dict, dotted: str, val):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = val


def apply_axis(cfg: dict, axis: str, v: float):
    """Special-case axes that aren't plain scalars."""
    if axis == "model.width":
        # two hidden layers of width v
        cfg["model"]["hidden_dims"] = [int(v), int(v)]
        return f"{int(v)}"
    set_nested(cfg, axis, v)
    return f"{v:g}"


def summarize(run_dir, lr: float) -> dict:
    R = load_run(run_dir)
    m = R["metrics"]
    final_loss = float(m["loss"].iloc[-1])
    diverged = not np.isfinite(final_loss)
    try:
        _, _, expl, _ = pca_project(R["W"], 2)
        pc1, pc2 = float(expl[0]), float(expl[1])
        pc12 = float(expl[:2].sum())
    except Exception:  # SVD fails on diverged (NaN) runs
        pc1 = pc2 = pc12 = float("nan")
    sh = R["sharp"]
    two_over_eta = 2.0 / lr
    if sh is not None and "lambda_max" in sh:
        lm = sh["lambda_max"].to_numpy(float)
        lm = lm[np.isfinite(lm)]
        if len(lm):
            eos_frac = float(np.mean(np.abs(lm - two_over_eta) <= 0.25 * two_over_eta))
            lm_init, lm_max, lm_final = float(lm[0]), float(lm.max()), float(lm[-1])
        else:
            eos_frac = lm_init = lm_max = lm_final = float("nan")
    else:
        eos_frac = lm_init = lm_max = lm_final = float("nan")
    try:
        res = delta_vtop_alignment(R)
        align_final = float(np.nanmean(res[1][-5:])) if res is not None else float("nan")
    except Exception:
        align_final = float("nan")
    return dict(
        lr=lr,
        final_loss=final_loss,
        min_loss=float(m["loss"].min()) if np.isfinite(m["loss"].min()) else float("nan"),
        pc1=pc1, pc2=pc2, pc12=pc12,
        lambda_max_init=lm_init, lambda_max_max=lm_max, lambda_max_final=lm_final,
        eos_frac=eos_frac, align_final=align_final, diverged=int(diverged),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--axis", default="optimizer.lr")
    ap.add_argument("--values", required=True, help="comma-separated floats")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--sweep-name", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--sharp-every", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="re-run even if variant already exists")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    if args.steps:
        base["train"]["steps"] = args.steps
    base["monitor"]["snapshot_every"] = args.snapshot_every
    if base.get("sharpness", {}).get("enabled"):
        base["sharpness"]["every"] = args.sharp_every

    sweep_name = args.sweep_name or f"sweep_{args.axis.replace('.', '_')}"
    sweep_dir = Path(args.run_root) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    values = [float(v) for v in args.values.split(",")]
    rows = []
    for v in values:
        cfg = copy.deepcopy(base)
        vsuf = apply_axis(cfg, args.axis, v)
        run_name = f"{sweep_name}_{vsuf}"
        cfg["run_name"] = run_name
        run_dir = sweep_dir / run_name
        if (not args.force and (run_dir / "snapshots.npz").exists()
                and (run_dir / "sharpness.csv").exists()):
            print(f"\n=== {args.axis} = {v}  ->  SKIP (exists: {run_dir}) ===", flush=True)
        else:
            print(f"\n=== {args.axis} = {v}  ->  {run_dir} ===", flush=True)
            Trainer(cfg, run_dir).train()
        lr_used = float(cfg["optimizer"]["lr"])
        s = summarize(run_dir, lr_used)
        s["axis_value"] = v
        s["variant"] = run_name
        rows.append(s)
        pd.DataFrame(rows).to_csv(sweep_dir / "summary.csv", index=False)

    print("\n================ SWEEP SUMMARY ================")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nsummary -> {sweep_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
