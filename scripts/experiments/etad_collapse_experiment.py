"""η×d COLLAPSE for the fixed-divisor causal control (review round 5).

QUESTION. The fixed-divisor control (fixedd_experiment.py) verified the 2-D
period-2 point prediction λ_eq = 2·d/η at a single η=0.05. A reviewer asks
whether the measured equilibrium really tracks λ_eq = 2d/η ACROSS η — i.e.
whether the invariant object is the dimensionless product η·λ_eq/2 = d. If the
late-window mean raw λmax collapses onto η·λ̄_late/2 ≈ d for every η (plotted
as η·λ̄/2 vs d falling on the y=x diagonal), the equilibrium is a property of
the (η, d) PAIR, not of one accidentally chosen learning rate.

GRID. η ∈ {0.025, 0.05, 0.1} × d ∈ {1.25, 1.5, 2.0, 2.5, 3.0}, n=5 seeds
(0–4), 3000 steps, identical protocol to fixedd_experiment.py (MLP [256,256]
tanh, teacher_student n=1024 dim=20 difficulty=0.3 teacher_hidden=[128,128],
data seed=1, --seed = model init only) except lr=η varies per arm. Cells that
diverge for ALL seeds are excluded from the collapse (they carry no equilibrium
to measure) and are shown in the phase diagram (48_fixedd_phase.pdf) instead.

Late-window convention matches analyze_fixedd.py: mean over measurements at
steps ≥ 1500 (i.e. 1500..2900 at MEASURE_EVERY=100); divergence = NaN in the
late λmax window or NaN/huge training loss.

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/etad_collapse_experiment.py --seed $s; done
    # round 13 (same-backend rerun incl. d=6.5, no prior-round mixing):
    for s in 0 1 2 3 4; do python scripts/experiments/etad_collapse_experiment.py --seed $s \
        --divisors 1.25 1.5 2.0 2.5 3.0 6.5 --source cpu_round13; done

Output:
    figures/etad_collapse_s{seed}.csv   one row per (η, d) cell, plus backend
                                        meta columns (device, torch_dtype,
                                        torch_version, timestamp, source)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # FORCE CPU — never contend with the GPU queue

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.optim import (
    SGD,
    EoSPrecondRankKState,
    eos_precond_rankk_step,
    register_param_names,
)
from steos.sharpness import max_eigenvalue

DEV = "cpu"
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100               # power-iteration sweeps for the ground-truth λmax
LATE_STEP = 1500           # late window: step >= 1500 (analyze_fixedd.py convention)
ETAS = [0.025, 0.05, 0.1]
DS = [1.25, 1.5, 2.0, 2.5, 3.0]
DIVERGENCE_LOSS = 1e6      # same threshold as fixedd_experiment.py


def opt_name(eta: float, d: float) -> str:
    return f"eta{eta:g}_d{d:g}"


def _new_model(task, seed: int = 0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def run_fixedd(eta: float, d: float, X, y, task, seed: int) -> tuple[dict, list]:
    """One (η, d, seed) fixed-divisor arm; returns (summary row, meas rows)."""
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    base = SGD(model.parameters(), lr=eta)
    register_param_names(base, model)
    st = EoSPrecondRankKState(1)
    losses: list[float] = []
    meas: list[tuple[int, float, float]] = []   # (step, λmax, λ̂₁)
    for step in range(STEPS):
        l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=eta, k=1,
                                   fixed_divisor=d)
        losses.append(l)
        if step % MEASURE_EVERY == 0:
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
            meas.append((step, lam, float(st.lam_hats[0])))

    lams_all = np.array([m[1] for m in meas], float)
    late = [m for m in meas if m[0] >= LATE_STEP]
    lams_late = np.array([m[1] for m in late], float)
    ok = lams_late[~np.isnan(lams_late)]
    lam_hat_late = np.array([m[2] for m in late], float)
    row = dict(
        eta=eta, d=d, opt=opt_name(eta, d),
        final_loss=float(losses[-1]),
        min_loss=float(min(losses)),
        lambda_max_final=float(lams_all[-1]),
        lambda_max_max=float(np.nanmax(lams_late)) if ok.size else float("nan"),
        diverged=bool(any(v != v for v in losses)
                      or losses[-1] > DIVERGENCE_LOSS),
        late_n_nan=int(np.isnan(lams_late).sum()),
        late_mean=float(ok.mean()) if ok.size else float("nan"),
        late_min=float(ok.min()) if ok.size else float("nan"),
        late_max=float(ok.max()) if ok.size else float("nan"),
        lam_hat1_late_mean=float(np.nanmean(lam_hat_late)),
    )
    return row, meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--etas", type=float, nargs="+", default=None,
                    help="subset of ETAS (debug); default all")
    ap.add_argument("--divisors", type=float, nargs="+", default=None,
                    help="subset of DS (default: all of DS, same as before)")
    ap.add_argument("--source", type=str, default="cpu_round13",
                    help="backend/source label recorded in every output row")
    ap.add_argument("--traj-out", type=str, default=None,
                    help="write step-level (eta,d,step,lambda_max) rows to this "
                         "CSV instead of figures/etad_collapse_s{seed}.csv "
                         "(trajectory-dump mode; never overwrites summaries)")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ETAD_THREADS", "8")))

    t0 = time.time()
    etas = args.etas if args.etas else ETAS
    ds = args.divisors if args.divisors else DS
    meta = dict(
        device=DEV,
        torch_dtype=str(torch.get_default_dtype()),
        torch_version=str(torch.__version__),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        source=args.source,
    )
    print(f"\n##### η×d COLLAPSE: etas={etas}, ds={ds}, steps={STEPS}, "
          f"DEV={DEV}, seed={args.seed}, threads={torch.get_num_threads()}, "
          f"backend={meta['torch_dtype']}/torch-{meta['torch_version']}, "
          f"source={meta['source']} #####",
          flush=True)
    torch.manual_seed(1)  # data seed — fixed (same as fixedd_experiment.py)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=(128, 128), device=DEV, seed=1)

    rows = []
    traj_rows = []          # only filled in --traj-out mode
    for eta in etas:
        for d in ds:
            ts = time.time()
            row, meas = run_fixedd(eta, d, X, y, task, seed=args.seed)
            if args.traj_out:
                traj_rows.extend(dict(eta=eta, d=d, opt=row["opt"],
                                      step=m[0], lambda_max=m[1],
                                      lam_hat1=m[2]) for m in meas)
            rows.append({**row, **meta})
            print(f"  {row['opt']:16s} ({time.time()-ts:5.1f}s): "
                  f"loss={row['final_loss']:.4f}  λ_late={row['late_mean']:7.2f}  "
                  f"pred 2d/η={2*d/eta:6.1f}  ηλ̄/2={eta*row['late_mean']/2 if row['late_mean']==row['late_mean'] else float('nan'):5.2f}"
                  f"{'  DIVERGED' if row['diverged'] or row['late_n_nan'] else ''}",
                  flush=True)

    Path("figures").mkdir(parents=True, exist_ok=True)
    if args.traj_out:
        tout = pd.DataFrame(traj_rows)
        tout.to_csv(args.traj_out, index=False)
        print(f"\n=> {args.traj_out}  ({len(tout)} measurement rows; "
              f"wall {time.time()-t0:.1f}s)")
        return

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== per-seed summary ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    out.to_csv(f"figures/etad_collapse_s{args.seed}.csv", index=False)
    print(f"\n=> figures/etad_collapse_s{args.seed}.csv  (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
