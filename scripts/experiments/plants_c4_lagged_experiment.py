"""C4 orthogonal-damping + lagged-phase plant experiment runner.

Both experiments run on the SAME tanh MLP [20-256-256-1] teacher-student full-
batch GD setup as ``rankk_sweep_experiment.py`` (LR=0.05, 2/η=40, 3000 steps,
n=1024, dim=20, difficulty=0.3, data seed=1 fixed, model seed parametrizes init).

Experiments:
  --exp c4:        orthogonal damping d ∈ {1, 1.5, 2, 4, 8}
                   Δθ = −η(g_∥ + (1/d) g_⊥); d=1 is vanilla SGD (sanity).
  --exp lagged:    lagged-phase plant k ∈ {0, 1, 2, 3}
                   k=0: accurate power iter (sharp-escape ~263, sanity)
                   k≥1: unit(Δθ_{t−k}) sensing (k=1 free-lagged pins ~40, sanity)

Per-step measurement (every step): loss, |x_t| = |(θ_t−θ_0)·u_1| (oscillation
amplitude along the INITIAL top eigenvector u_1).
Every MEASURE_LAM_EVERY steps: full-batch λ_max (warm-started power iter).

Outputs:
  runs/{c4_orth_d{D}, lagged_k{K}}_s{SEED}/metrics.csv  (one row per measurement)
  runs/{c4_orth_d{D}, lagged_k{K}}_s{SEED}/per_step.csv (one row per step: loss, |x_t|)
  runs/{c4_orth_d{D}, lagged_k{K}}_s{SEED}/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.optim import (
    SGD, register_param_names,
    OrthDampState, orth_damp_step,
    LaggedPlantState, lagged_plant_step,
)
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
STEPS = 3000
MEASURE_LAM_EVERY = 50        # full λ_max measurement cadence
N_ITER = 100                  # power-iter sweeps for ground-truth λ_max
TWO_OVER_ETA = 2.0 / LR       # = 40
# Divergence criteria. The literal "loss > init_loss" triggers spuriously on
# normal EoS period-2 oscillation (loss can bounce to ~3-4x init mid-training
# even when the run converges fine). Use DURABLE criteria instead:
#   diverged_final  : final_loss > init_loss  (run ended worse than it started)
#   diverged_lam    : peak λ_max > 10×(2/η) = 400  (spectral explosion)
DIVERGE_LAM_MULT = 10.0


def _new_model(task, seed):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def _flatten(params):
    return torch.cat([p.reshape(-1) for p in params])


def run_one(exp_kind, param_value, X, y, task, seed, no_clamp=False,
            sensing_mode="dtheta"):
    """Run one (experiment, parameter, seed) and return (summ_dict, per_step_df, measure_df).

    measure_df: rows at every MEASURE_LAM_EVERY step + final step (λ_max, loss, |x_t|, diverged).
    per_step_df: rows at every step (loss, |x_t|) — high-res oscillation data.
    """
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    base = SGD(model.parameters(), lr=LR)
    register_param_names(base, model)
    params = [p for p in model.parameters() if p.requires_grad]

    # compute u_1 = top eigenvector at θ_0 (used for |x_t| oscillation amplitude)
    lam_init, u1_list = max_eigenvalue(model, loss, X, y, n_iter=N_ITER,
                                        tol=1e-6, return_vec=True)
    u1_flat = _flatten([u.detach() for u in u1_list])
    theta0_flat = _flatten([p.detach().clone() for p in params])

    # state for the chosen controller
    if exp_kind == "c4":
        state = OrthDampState(n_sweeps=3)
    elif exp_kind == "lagged":
        state = LaggedPlantState(lag_k=int(param_value), accurate_iters=80,
                                  accurate_tol=1e-4,
                                  sensing_mode=sensing_mode)
    else:
        raise ValueError(exp_kind)

    init_loss = float(loss(model(X), y).item())
    per_step_rows = []
    measure_rows = []
    lam_max_seen = float("-inf")

    def _record(step, l, do_lam=False):
        nonlocal lam_max_seen
        with torch.no_grad():
            theta_t_flat = _flatten([p.detach().clone() for p in params])
            disp = float((theta_t_flat - theta0_flat).dot(u1_flat).item())
            x_t = abs(disp)
        lam = float("nan")
        if do_lam:
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER,
                                        tol=1e-6))
            lam_max_seen = max(lam_max_seen, lam)
        return dict(step=step, loss=l, x_t=x_t, disp_signed=disp,
                    lambda_max=lam)

    # initial measurement (step 0, before any update)
    measure_rows.append(_record(0, init_loss, do_lam=True))

    for step in range(STEPS):
        if exp_kind == "c4":
            l = orth_damp_step(model, base, loss, X, y, state, lr=LR,
                                d=float(param_value))
        else:
            l = lagged_plant_step(model, base, loss, X, y, state, lr=LR,
                                  clamp=not no_clamp)
        do_lam = ((step + 1) % MEASURE_LAM_EVERY == 0) or (step == STEPS - 1)
        # always record per-step (cheap)
        with torch.no_grad():
            theta_t_flat = _flatten([p.detach().clone() for p in params])
            disp = float((theta_t_flat - theta0_flat).dot(u1_flat).item())
        per_step_rows.append(dict(step=step + 1, loss=l, x_t=abs(disp),
                                    disp_signed=disp))
        if do_lam:
            measure_rows.append(_record(step + 1, l, do_lam=True))

    measure_df = pd.DataFrame(measure_rows)
    per_step_df = pd.DataFrame(per_step_rows)

    # per-step oscillation amplitude stats (after step 500 — skip init transient)
    late_xt = per_step_df[per_step_df.step >= 500].x_t
    xt_mean = float(late_xt.mean()) if len(late_xt) else float("nan")
    xt_max = float(late_xt.max()) if len(late_xt) else float("nan")
    # oscillation = max − min of x_t over a sliding window of 50 steps
    if len(late_xt) >= 50:
        rolling = per_step_df[per_step_df.step >= 500].x_t.rolling(50, min_periods=1)
        osc = float((rolling.max() - rolling.min()).mean())
    else:
        osc = float("nan")

    last_lam = float(measure_df.lambda_max.iloc[-1])
    last_loss = float(measure_df.loss.iloc[-1])
    peak_lam = float(measure_df.lambda_max.max())
    # DURABLE divergence criteria (see comment near DIVERGE_LAM_MULT):
    #   diverged_final : run ended worse than it started (permanent divergence)
    #   diverged_lam   : spectral explosion (peak λ_max > 10×2/η = 400)
    diverged_final = bool(last_loss > init_loss)
    diverged_lam = bool(peak_lam > DIVERGE_LAM_MULT * TWO_OVER_ETA)
    diverged = bool(diverged_final or diverged_lam)
    summary = dict(
        exp=exp_kind, param=param_value, seed=seed,
        lambda_max_init=float(lam_init),
        lambda_max_final=last_lam,
        lambda_max_peak=peak_lam,
        loss_init=init_loss,
        loss_final=last_loss,
        loss_min=float(per_step_df.loss.min()),
        xt_step500_mean=xt_mean,
        xt_step500_max=xt_max,
        xt_oscillation_window50=osc,
        diverged=int(diverged),
        diverged_final=int(diverged_final),
        diverged_lam=int(diverged_lam),
    )
    return summary, per_step_df, measure_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["c4", "lagged"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clamp", action="store_true",
                    help="lagged: use clamp-free rank-1 formula (cap-removal). "
                         "Default OFF (clamp=True) matches existing accurate / "
                         "free-lagged variants and is REQUIRED for k≥1 stability.")
    ap.add_argument("--sensing", choices=["dtheta", "eigvec"], default="dtheta",
                    help="lagged: sensing vector source. 'dtheta' (default) = "
                         "unit(Δθ_{t−k}) — free-lagged (confounds lag+inaccuracy). "
                         "'eigvec' = û(t−k), accurate power-iter eigenvector from "
                         "k steps ago — ISOLATES lag from inaccuracy.")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(1)   # data seed fixed (matches rankk_sweep_experiment.py)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                                teacher_hidden=[128, 128], device=DEV, seed=1)

    # determine parameter value from env (the .sh driver exports D / K)
    if args.exp == "c4":
        param_env = "D_VALUE"
        param_value = float(__import__("os").environ[param_env])
        run_name = f"c4_orth_d{param_value}_s{args.seed}"
    else:
        param_env = "K_VALUE"
        param_value = int(__import__("os").environ[param_env])
        # run-name encodes sensing mode: dtheta → lagged_k{K}, eigvec → lagged_eig_k{K}
        if args.sensing == "eigvec":
            run_name = f"lagged_eig_k{param_value}_s{args.seed}"
        else:
            run_name = f"lagged_k{param_value}_s{args.seed}"

    RUN_DIR = Path("runs") / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n##### {args.exp} param={param_value} seed={args.seed} "
          f"η={LR} steps={STEPS} 2/η={TWO_OVER_ETA:g} DEV={DEV} #####", flush=True)

    summary, per_step_df, measure_df = run_one(
        args.exp, param_value, X, y, task, seed=args.seed,
        no_clamp=args.no_clamp, sensing_mode=args.sensing)

    measure_df.to_csv(RUN_DIR / "metrics.csv", index=False)
    per_step_df.to_csv(RUN_DIR / "per_step.csv", index=False)
    (RUN_DIR / "manifest.json").write_text(json.dumps({
        **summary, "exp": args.exp, "lr": LR, "steps": STEPS,
        "two_over_eta": TWO_OVER_ETA,
        "measure_lam_every": MEASURE_LAM_EVERY,
        "n_iter_lambda": N_ITER,
        "diverge_lam_mult": DIVERGE_LAM_MULT,
        "total_time_s": time.time() - t0,
    }, indent=2))

    print(f"  λ_max init={summary['lambda_max_init']:.2f} "
          f"final={summary['lambda_max_final']:.2f} "
          f"peak={summary['lambda_max_peak']:.2f}")
    print(f"  loss init={summary['loss_init']:.4f} final={summary['loss_final']:.4f} "
          f"min={summary['loss_min']:.4f}")
    print(f"  |x_t| mean(≥500)={summary['xt_step500_mean']:.4f} "
          f"max={summary['xt_step500_max']:.4f} "
          f"osc={summary['xt_oscillation_window50']:.4f}")
    print(f"  diverged={bool(summary['diverged'])} "
          f"(final={bool(summary['diverged_final'])} "
          f"lam_peak={bool(summary['diverged_lam'])})")
    print(f"=> {RUN_DIR} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
