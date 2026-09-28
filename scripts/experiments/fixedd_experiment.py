"""Fixed-divisor rank-1 damping — the causal control that de-confounds the
preconditioned-sharpness invariant (review round 4; Major Concern 1).

PROBLEM BEING FIXED. The accurate rank-1 damping arm uses the self-gating
divisor d = max(λ̂₁/(2/η), 1). Whenever the clamp fires, the "invariant"
λ̂₁/d = 2/η holds to machine precision BY CONSTRUCTION — the divisor is
λ̂₁/(2/η) by definition, so dividing it back out is an algebraic identity of
the controller, not a dynamical property of the optimizer. The Fig-34 panel
that shows λ̂₁/d pinned at 2/η "to machine precision" therefore carries no
causal information about what the EoS dynamics regulate.

THE CAUSAL CONTROL. Remove the feedback: hold the divisor FIXED at d_fix
(the edit is still the same damping g' = g − (1 − 1/d_fix)(gᵀv̂)v̂, sensed by
the identical warm-started block power iteration, applied unconditionally
every step). With γ = 1 − 1/d_fix constant, the 2-D period-2 fixed point
y* = 2γ/((1−γ)η) is no longer vacuous (with the adaptive rule, substituting
γ = 1 − 2/(ηλ̂₁) reduces the fixed-point condition to the identity y* = y).
It makes a non-trivial POINT PREDICTION for the RAW equilibrium:

    λ_eq(d_fix) = 2·d_fix/η
      d_fix = 1.25 → 50,  1.5 → 60,  2 → 80,  3 → 120,  6.5 → 260   (η=0.05)

If the measured late raw λ_max tracks 2·d_fix/η across the grid — in
particular if d_fix = 6.5 reproduces the adaptive arm's ~262 equilibrium
(the adaptive arm's realized mean divisor was d̄ = 6.55) WITHOUT any feedback
— then "EoS regulates the preconditioned sharpness λ_eff = λ₁/d" is a causal
property of the dynamics, not a controller identity. If it does not track,
the invariant claim must be weakened; either way the experiment is decisive.

Protocol is IDENTICAL to ``rankk_sweep_experiment.py`` /
``amplify_rank1_experiment.py`` (LR=0.05 ⇒ 2/η=40, 3000 steps, MLP [256,256]
tanh, teacher_student dim=20 difficulty=0.3 teacher_hidden=[128,128], data
seed=1 fixed, ``--seed`` parametrizes model-init only), so the SGD reference
(λ_max_final ≈ 40) and the adaptive damping reference (≈ 262) are directly
comparable (both are re-run in-session for a same-environment table).

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/fixedd_experiment.py --seed $s; done

Outputs:
    figures/fixedd_summary_s{seed}.csv  per-arm summary (as rankk/amplify)
    figures/fixedd_traj_s{seed}.csv     long: opt, step, lambda_max, lam_hat1
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
)
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100             # power-iteration sweeps for the ground-truth λ_max
TWO_OVER_ETA = 2.0 / LR  # = 40
FIXED_DS = [1.25, 1.5, 2.0, 2.5, 3.0, 6.5]  # predicted equilibria 2d/η = 50…260
# 6.5 anchors the grid on the adaptive arm's realized mean divisor
# (d̄ = 261.8/40 = 6.55), so fixed-d 6.5 vs the adaptive outcome is an
# apples-to-apples with/without-feedback comparison.


def opt_name(d: float) -> str:
    return f"fixedd_d{d:g}"


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
    ctrl_state: EoSPrecondRankKState | None = None

    def measure(step: int):
        lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
        lam_hat1 = (ctrl_state.lam_hats[0] if ctrl_state is not None
                    else float("nan"))
        traj.append((opt, step, lam, lam_hat1))
        return lam

    lam_true_vals: list[float] = []
    if opt.startswith("fixedd_d"):
        d = float(opt.split("_d")[1])
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = EoSPrecondRankKState(1)
        ctrl_state = st
        for step in range(STEPS):
            l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=1,
                                       fixed_divisor=d)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true_vals.append(measure(step))
    elif opt == "rankk_k1":  # adaptive-damping reference, re-run in-session
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = EoSPrecondRankKState(1)
        ctrl_state = st
        for step in range(STEPS):
            l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=1)
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
        diverged=bool(any(x != x for x in losses) or losses[-1] > 1e6),
    )
    return summary, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    print(f"\n##### FIXED-DIVISOR rank-1 damping: η={LR}, steps={STEPS}, "
          f"2/η={TWO_OVER_ETA:g}, DEV={DEV}, seed={args.seed} #####")
    torch.manual_seed(1)  # data seed — fixed (same as rankk_sweep_experiment.py)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)

    opt_names = ["sgd", "rankk_k1"] + [opt_name(d) for d in FIXED_DS]
    summary_rows, all_traj = [], []
    for name in opt_names:
        ts = time.time()
        summ, traj = run(name, X, y, task, seed=args.seed)
        summary_rows.append(summ)
        all_traj.extend(traj)
        print(f"  {name:12s} ({time.time()-ts:5.1f}s): loss={summ['final_loss']:.4f}  "
              f"λ_max_final={summ['lambda_max_final']:6.2f}  "
              f"λ_max_peak={summ['lambda_max_max']:8.1f}"
              f"{'  DIVERGED' if summ['diverged'] else ''}", flush=True)

    summ = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== summary ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== VERDICT vs the 2-D point prediction λ_eq = 2·d_fix/η ===")
    print("  (fixed γ = 1 − 1/d: y* = 2γ/((1−γ)η) is NOT vacuous, unlike the"
          " adaptive rule)")
    for d in FIXED_DS:
        row = summ[summ.opt == opt_name(d)].iloc[0]
        pred = TWO_OVER_ETA * d
        ratio = row["lambda_max_final"] / pred if pred else float("nan")
        print(f"  d={d:<4g}: λ_max_final={row['lambda_max_final']:7.2f}  "
              f"predicted={pred:6.1f}  ratio={ratio:5.2f}"
              f"{'  DIVERGED' if row['diverged'] else ''}")
    for ref in ("sgd", "rankk_k1"):
        row = summ[summ.opt == ref].iloc[0]
        print(f"  {ref:10s}: λ_max_final={row['lambda_max_final']:7.2f}")

    Path("figures").mkdir(parents=True, exist_ok=True)
    summ.to_csv(f"figures/fixedd_summary_s{args.seed}.csv", index=False)
    pd.DataFrame(all_traj, columns=["opt", "step", "lambda_max", "lam_hat1"]).to_csv(
        f"figures/fixedd_traj_s{args.seed}.csv", index=False)
    print(f"\n=> figures/fixedd_summary_s{args.seed}.csv "
          f"+ figures/fixedd_traj_s{args.seed}.csv  (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
