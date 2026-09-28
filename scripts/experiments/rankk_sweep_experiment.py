"""Rank-k curvature-preconditioning sweep — does damping the top-k Hessian
eigenvectors let the optimizer escape the EoS threshold 2/η?

Generalizes the rank-1 EoS-Precond falsification (``eos_precond_experiment.py``)
to k ∈ {1, 2, 5, 10}. Prediction (Claim 1 generalized): NO — for k ≪ n,
sharpness migrates to the undamped complement and λ_max re-pins at 2/η. Only
SAM (a global plant action over ALL directions) escapes. The top-12 Hessian
spectrum is recorded at each measurement step so we can VISUALIZE the migration
(sharpness spreading to λ_2, …, λ_k when the top-k are damped).

Setup is IDENTICAL to ``eos_precond_experiment.py`` (LR=0.05, STEPS=3000,
MEASURE_EVERY=100, N_ITER=100, MLP [256, 256] tanh, teacher_student dim=20
difficulty=0.3 teacher_hidden=[128, 128], data seed=1 fixed, ``--seed``
parametrizes model-init only) so the SGD baseline reproduces λ_max_final ≈ 40.

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/rankk_sweep_experiment.py --seed $s; done

Outputs:
    figures/rankk_summary_s{seed}.csv    per-opt: opt, final_loss, min_loss,
                                         lambda_max_final, lambda_max_max
    figures/rankk_spectrum_s{seed}.csv   long: opt, step, idx, lambda_i (top-12)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
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
from steos.sharpness import max_eigenvalue, top_k_eigenvalues

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100            # power-iteration sweeps for the ground-truth λ_max
SPEC_K = 12             # top-12 eigenvalues for the migration spectrum
SPEC_ITER = 40          # block-power sweeps per spectrum measurement (flat-MGS ⇒ fast)
TWO_OVER_ETA = 2.0 / LR  # = 40
RANKK_KS = [1, 2, 5, 10]
RHO = 0.05              # SAM perturbation radius (matches eos_precond_experiment.py)
# the full top-12 spectrum is recorded only for the rank-k variants (the
# migration diagnostic); SGD/SAM don't need it for the figure and it is ~2× the
# optimizer cost to measure, so skipping it halves SGD/SAM wall-clock.
SPEC_OPTS = {"sgd": False, "sam": False}
for _k in RANKK_KS:
    SPEC_OPTS[f"rankk_k{_k}"] = True


def _new_model(task, seed: int = 0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def _measure(loss, model, X, y, opt_name: str):
    """Ground-truth λ_max (power iter, tol-stopped) always; + top-12 spectrum
    only for the rank-k variants (the migration diagnostic)."""
    lam_true = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
    specs: Optional[List[float]] = None
    if SPEC_OPTS.get(opt_name, False):
        r = top_k_eigenvalues(model, loss, X, y, k=SPEC_K, n_iter=SPEC_ITER)
        specs = r if isinstance(r, list) else None
    return lam_true, specs


def run(opt_name, X, y, task, seed: int = 0):
    """Run one optimizer.

    Returns:
        summary: dict(opt, final_loss, min_loss, lambda_max_final, lambda_max_max)
        spectrum_rows: list of (opt, step, idx, lambda_i) at each measurement step
                       (empty for SGD/SAM — spectrum not recorded).
    """
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    losses = []
    lam_true_vals = []          # ground-truth λ_max at measurement steps only
    spectrum_rows = []
    ctrl_rows = []              # controller-side (λ̂_i, d_i, cos) at measurement steps

    if opt_name.startswith("rankk_k"):
        k = int(opt_name.split("_k")[1])
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = EoSPrecondRankKState(k)
        for step in range(STEPS):
            l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=k)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true, specs = _measure(loss, model, X, y, opt_name)
                lam_true_vals.append(lam_true)
                if specs is not None:
                    for idx, li in enumerate(specs):
                        spectrum_rows.append((opt_name, step, idx, float(li)))
                # controller-side quantities for the preconditioned-sharpness
                # (λ_eff = λ̂/d) analysis: the sensed Rayleigh λ̂_i, the
                # self-gating divisor d_i, and the resulting effective
                # (preconditioned) sharpness the 2/η boundary actually governs
                for i in range(k):
                    lam_h = st.lam_hats[i]
                    d_i = st.divisors[i]
                    ctrl_rows.append((opt_name, step, i, float(lam_h), float(d_i),
                                      float(st.cos_g_v[i])))
    elif opt_name == "sam":
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        for step in range(STEPS):
            l = sam_step(model, base, loss, X, y, rho=RHO)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true, specs = _measure(loss, model, X, y, opt_name)
                lam_true_vals.append(lam_true)
                if specs is not None:
                    for idx, li in enumerate(specs):
                        spectrum_rows.append((opt_name, step, idx, float(li)))
    elif opt_name == "sgd":
        opt = SGD(model.parameters(), lr=LR)
        register_param_names(opt, model)
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True)
            out = model(X)
            l = float(loss(out, y).item())
            loss(out, y).backward()
            opt.step()
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true, specs = _measure(loss, model, X, y, opt_name)
                lam_true_vals.append(lam_true)
                if specs is not None:
                    for idx, li in enumerate(specs):
                        spectrum_rows.append((opt_name, step, idx, float(li)))
    else:
        raise ValueError(opt_name)

    summary = dict(
        opt=opt_name,
        final_loss=float(losses[-1]) if losses else float("nan"),
        min_loss=float(min(losses)) if losses else float("nan"),
        lambda_max_final=float(lam_true_vals[-1]) if lam_true_vals else float("nan"),
        lambda_max_max=float(max(lam_true_vals)) if lam_true_vals else float("nan"),
    )
    return summary, spectrum_rows, ctrl_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    print(f"\n##### Rank-k sweep: η={LR}, steps={STEPS}, 2/η={TWO_OVER_ETA:g}, "
          f"DEV={DEV}, seed={args.seed} #####")
    torch.manual_seed(1)  # data seed — fixed so every model-init seed sees identical data
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)

    opt_names = ["sgd", "sam"] + [f"rankk_k{k}" for k in RANKK_KS]
    summary_rows, all_spectrum, all_ctrl = [], [], []
    for name in opt_names:
        ts = time.time()
        summ, spec, ctrl = run(name, X, y, task, seed=args.seed)
        summary_rows.append(summ)
        all_spectrum.extend(spec)
        all_ctrl.extend(ctrl)
        print(f"  {name:10s} ({time.time()-ts:5.1f}s): loss={summ['final_loss']:.4f}  "
              f"λ_max_final={summ['lambda_max_final']:6.2f}  "
              f"λ_max_peak={summ['lambda_max_max']:.1f}", flush=True)

    summ = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== summary ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- verdict ----------------------------------------------------------
    lam_thresh = 0.7 * TWO_OVER_ETA
    print(f"\n=== VERDICT (escape needs λ_max_final < {lam_thresh:.1f} = 0.7·2/η) ===")
    for k in RANKK_KS:
        name = f"rankk_k{k}"
        row = summ[summ.opt == name].iloc[0]
        escaped = row["lambda_max_final"] < lam_thresh
        print(f"  rank-k k={k:<2d}: λ_max_final={row['lambda_max_final']:6.2f}  "
              f"{'ESCAPE' if escaped else 'LOCK at 2/η'}")
    for ref in ("sgd", "sam"):
        row = summ[summ.opt == ref].iloc[0]
        print(f"  {ref:8s}: λ_max_final={row['lambda_max_final']:6.2f}")

    Path("figures").mkdir(parents=True, exist_ok=True)
    summ.to_csv(f"figures/rankk_summary_s{args.seed}.csv", index=False)
    pd.DataFrame(all_spectrum, columns=["opt", "step", "idx", "lambda_i"]).to_csv(
        f"figures/rankk_spectrum_s{args.seed}.csv", index=False)
    if all_ctrl:
        pd.DataFrame(all_ctrl, columns=["opt", "step", "i", "lam_hat", "divisor",
                                        "cos_g_v"]).to_csv(
            f"figures/rankk_ctrl_s{args.seed}.csv", index=False)
    print(f"\n=> figures/rankk_summary_s{args.seed}.csv "
          f"+ figures/rankk_spectrum_s{args.seed}.csv  (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
