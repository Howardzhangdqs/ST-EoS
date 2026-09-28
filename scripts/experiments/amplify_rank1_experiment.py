"""Rank-1 curvature-AMPLIFICATION control — the sign-flip test of
Proposition ``prop:rank1-no-flat-escape``.

The paper's rank-1 plant DAMPS the gradient along the top eigenvector
(g' = g − γ(gᵀv̂)v̂, γ ∈ (0,1)) and the 2-D reduction predicts (and the rank-k
sweep confirms) the equilibrium moves UP: y* = 2γ/((1−γ)η) > 0, sharp-escape to
λ ≈ 262. The same algebra at γ = −α < 0 (AMPLIFICATION: g' = g + α(gᵀv̂)v̂)
predicts the equilibrium moves DOWN: y* = −2α/((1+α)η) < 0, i.e. the period-2
equilibrium descends to

    λ* = 2/((1+α)η)     (α=0.25 → 32.0, α=0.5 → 26.67, α=1 → 20.0)

— flat escape by a LOCAL, rank-1, single-direction actuator. This experiment
runs that control on the same protocol as ``rankk_sweep_experiment.py``
(LR=0.05 ⇒ 2/η=40, 3000 steps, MLP [256,256] tanh, teacher_student dim=20
difficulty=0.3 teacher_hidden=[128,128], data seed=1 fixed, ``--seed``
parametrizes model-init only) so SGD / SAM / rankk_k1-damping references from
that sweep are directly comparable (all three are re-run in-session anyway for
a same-environment table).

Sensing: identical warm-started block power iteration as the damping arm
(k=1, n_sweeps=3 ⇒ ~3 HVPs/step + 1 HVP edit) — the only difference is the
SIGN of the edit. If λ_max settles at 2/((1+α)η), the theory's sign map
(damping ⇔ up, amplification ⇔ down) is confirmed and the paper's dichotomy
becomes attenuation-vs-amplification rather than local-vs-global.

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/amplify_rank1_experiment.py --seed $s; done

Outputs:
    figures/amplify_summary_s{seed}.csv   per-opt summary (as rankk)
    figures/amplify_traj_s{seed}.csv      long: opt, step, lambda_max
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.optim import (
    SGD,
    EoSPrecondRankKState,
    eos_precond_rankk_step,
    register_param_names,
    sam_step,
)
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100             # power-iteration sweeps for the ground-truth λ_max
TWO_OVER_ETA = 2.0 / LR  # = 40
AMPS = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]  # γ = −α: predicted equilibria 2/((1+α)η); 0.05/0.1/0.2 added 2026-09-21 for the weak-edit-limit verification (writer Issue 8)
RHO = 0.05               # SAM perturbation radius (matches rankk_sweep_experiment.py)


def opt_name(a: float) -> str:
    return f"amp_a{a:g}"


def _new_model(task, seed: int = 0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def run(opt, X, y, task, seed: int = 0):
    """Run one optimizer arm; returns (summary, trajectory rows)."""
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    losses: list[float] = []
    traj: list[tuple] = []

    def measure(step: int):
        lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
        traj.append((opt, step, lam))
        return lam

    lam_true_vals: list[float] = []
    if opt.startswith("amp_a"):
        a = float(opt.split("_a")[1])
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = EoSPrecondRankKState(1)
        for step in range(STEPS):
            l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=1,
                                       amplify=a)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true_vals.append(measure(step))
    elif opt == "rankk_k1":  # the damping reference, re-run in-session
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = EoSPrecondRankKState(1)
        for step in range(STEPS):
            l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=1)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true_vals.append(measure(step))
    elif opt == "sam":
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        for step in range(STEPS):
            l = sam_step(model, base, loss, X, y, rho=RHO)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true_vals.append(measure(step))
    elif opt == "sgd":
        o = SGD(model.parameters(), lr=LR)
        register_param_names(o, model)
        for step in range(STEPS):
            o.zero_grad(set_to_none=True)
            out = model(X)
            l = float(loss(out, y).item())
            loss(out, y).backward()
            o.step()
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true_vals.append(measure(step))
    else:
        raise ValueError(opt)

    summary = dict(
        opt=opt,
        final_loss=float(losses[-1]) if losses else float("nan"),
        min_loss=float(min(losses)) if losses else float("nan"),
        lambda_max_final=float(lam_true_vals[-1]) if lam_true_vals else float("nan"),
        lambda_max_max=float(max(lam_true_vals)) if lam_true_vals else float("nan"),
    )
    return summary, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    print(f"\n##### Rank-1 AMPLIFICATION control: η={LR}, steps={STEPS}, "
          f"2/η={TWO_OVER_ETA:g}, DEV={DEV}, seed={args.seed} #####")
    torch.manual_seed(1)  # data seed — fixed (same as rankk_sweep_experiment.py)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)

    opt_names = (["sgd", "sam", "rankk_k1"] + [opt_name(a) for a in AMPS])
    summary_rows, all_traj = [], []
    for name in opt_names:
        ts = time.time()
        summ, traj = run(name, X, y, task, seed=args.seed)
        summary_rows.append(summ)
        all_traj.extend(traj)
        print(f"  {name:12s} ({time.time()-ts:5.1f}s): loss={summ['final_loss']:.4f}  "
              f"λ_max_final={summ['lambda_max_final']:6.2f}  "
              f"λ_max_peak={summ['lambda_max_max']:8.1f}", flush=True)

    summ = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== summary ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== VERDICT vs the 2-D sign map (amplification ⇒ λ* = 2/((1+α)η)) ===")
    for a in AMPS:
        row = summ[summ.opt == opt_name(a)].iloc[0]
        pred = TWO_OVER_ETA / (1.0 + a)
        escaped = row["lambda_max_final"] < 0.9 * TWO_OVER_ETA  # clearly below the SGD pin
        print(f"  α={a:<4g}: λ_max_final={row['lambda_max_final']:6.2f}  "
              f"predicted λ*={pred:6.2f}  "
              f"{'FLAT-ESCAPE' if escaped else 'no escape'}")
    for ref in ("sgd", "sam", "rankk_k1"):
        row = summ[summ.opt == ref].iloc[0]
        print(f"  {ref:10s}: λ_max_final={row['lambda_max_final']:6.2f}")

    Path("figures").mkdir(parents=True, exist_ok=True)
    summ.to_csv(f"figures/amplify_summary_s{args.seed}.csv", index=False)
    pd.DataFrame(all_traj, columns=["opt", "step", "lambda_max"]).to_csv(
        f"figures/amplify_traj_s{args.seed}.csv", index=False)
    print(f"\n=> figures/amplify_summary_s{args.seed}.csv "
          f"+ figures/amplify_traj_s{args.seed}.csv  (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
