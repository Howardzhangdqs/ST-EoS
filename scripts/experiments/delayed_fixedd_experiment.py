"""DELAYED-ONSET fixed-d control (review round 5) — de-confounds TIMING from
FEEDBACK in the fixed-divisor instability.

CONFOUND. At η=0.05 the adaptive rule (self-gating divisor
d = max(λ̂₁/(2/η), 1)) is 5/5 stable, while fixed d ∈ {2.5, 3.0} applied from
step 0 diverges 5/5 (fixedd_experiment.py). Two explanations:
  (a) TIMING / warm-up: damping a direction whose sharpness is still below the
      EoS boundary from step 0 is destructive — the adaptive rule simply waits
      (its gate is closed while λ̂₁ < 2/η), and ANY controller that waits would
      be stable;
  (b) FEEDBACK: the λ̂-dependence of the adaptive divisor is itself what
      stabilizes the orbit, regardless of when damping starts.

CONTROL. Per seed: run the adaptive arm once and record T_gate = the first step
at which its gate actually OPENS (st.n_fired > 0 ⇒ λ̂₁ > 2/η). Then run a
DELAYED arm: plain GD (no damping, no sensing) for steps < T_gate, then the
FIXED divisor d ∈ {2.5, 3.0} for steps ≥ T_gate, to step 3000.

WHY THE WRAPPER IS EXACT. While the gate is closed the adaptive step applies
NO edit (divisor 1 ⇒ c = α(1−1/1) = 0) and is arithmetically identical to a
plain GD step, so the delayed arm reproduces the adaptive arm's parameter
trajectory bit-for-bit up to T_gate; the arms then branch (adaptive feedback vs
constant d) from a common state. Implemented entirely in this script — no
steos/ changes.

READ-OUT. If the delayed arms survive (λ tracks ≈ 2d/η like d ≤ 2 fixed arms),
stability is a TIMING/warm-up property. If they still diverge, the adaptive
FEEDBACK is genuinely stabilizing (a constant divisor cannot hold the orbit
even past the natural gate-opening time).

Protocol identical to fixedd_experiment.py at η=0.05 (MLP [256,256] tanh,
teacher_student n=1024 dim=20 difficulty=0.3 teacher_hidden=[128,128], data
seed=1, --seed = model init only). Late window step ≥ 1500 (analyze_fixedd.py).

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/delayed_fixedd_experiment.py --seed $s; done

Output:
    figures/delayed_fixedd_s{seed}.csv   rows: adaptive, delayed_d2.5, delayed_d3
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
LR = 0.05
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100
LATE_STEP = 1500
TWO_OVER_ETA = 2.0 / LR   # 40 — gate opens when λ̂₁ crosses this
DELAYED_DS = [2.5, 3.0]
DIVERGENCE_LOSS = 1e6


def _new_model(task, seed: int = 0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def _summarize(arm: str, losses, meas, t_gate: int) -> dict:
    lams_all = np.array([m[1] for m in meas], float)
    late = [m for m in meas if m[0] >= LATE_STEP]
    lams_late = np.array([m[1] for m in late], float)
    ok = lams_late[~np.isnan(lams_late)]
    lam_hat_late = np.array([m[2] for m in late], float)
    return dict(
        arm=arm, T_gate=t_gate,
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


def run_adaptive(X, y, task, seed: int) -> tuple[dict, int]:
    """Adaptive (self-gating) reference arm; also returns T_gate = first step
    where the damping gate actually opens (n_fired > 0), or STEPS if never."""
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    base = SGD(model.parameters(), lr=LR)
    register_param_names(base, model)
    st = EoSPrecondRankKState(1)
    losses: list[float] = []
    meas: list[tuple[int, float, float]] = []
    t_gate = STEPS
    for step in range(STEPS):
        l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=1)
        losses.append(l)
        if st.n_fired > 0 and t_gate == STEPS and step < STEPS:
            # first open — but only count it if the gate had been closed before
            t_gate = step
        if step % MEASURE_EVERY == 0:
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
            meas.append((step, lam, float(st.lam_hats[0])))
    return _summarize("rankk_k1", losses, meas, t_gate), t_gate


def run_delayed(d: float, t_gate: int, X, y, task, seed: int) -> dict:
    """Plain GD for steps < t_gate, then fixed-divisor d for steps ≥ t_gate."""
    model = _new_model(task, seed=seed)   # same init as the adaptive arm
    loss = loss_fn(task)
    base = SGD(model.parameters(), lr=LR)
    register_param_names(base, model)
    st = EoSPrecondRankKState(1)
    losses: list[float] = []
    meas: list[tuple[int, float, float]] = []
    for step in range(STEPS):
        if step < t_gate:
            base.zero_grad(set_to_none=True)
            out = model(X)
            l = float(loss(out, y).item())
            loss(out, y).backward()
            base.step()
        else:
            l = eos_precond_rankk_step(model, base, loss, X, y, st, lr=LR, k=1,
                                       fixed_divisor=d)
        losses.append(l)
        if step % MEASURE_EVERY == 0:
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
            lam_hat1 = float(st.lam_hats[0]) if st.V is not None else float("nan")
            meas.append((step, lam, lam_hat1))
    return _summarize(f"delayed_d{d:g}", losses, meas, t_gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ETAD_THREADS", "8")))

    t0 = time.time()
    print(f"\n##### DELAYED-ONSET fixed-d: η={LR}, 2/η={TWO_OVER_ETA:g}, "
          f"ds={DELAYED_DS}, steps={STEPS}, DEV={DEV}, seed={args.seed}, "
          f"threads={torch.get_num_threads()} #####", flush=True)
    torch.manual_seed(1)  # data seed — fixed
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=(128, 128), device=DEV, seed=1)

    rows = []
    ts = time.time()
    adaptive_row, t_gate = run_adaptive(X, y, task, seed=args.seed)
    rows.append(adaptive_row)
    print(f"  adaptive ({time.time()-ts:5.1f}s): T_gate={t_gate}  "
          f"loss={adaptive_row['final_loss']:.4f}  "
          f"λ_late={adaptive_row['late_mean']:.2f}"
          f"{'  DIVERGED' if adaptive_row['diverged'] else ''}", flush=True)
    if t_gate >= STEPS:
        print("  WARNING: gate never opened within 3000 steps — "
              "delayed arms degenerate to plain GD.", flush=True)
    for d in DELAYED_DS:
        ts = time.time()
        row = run_delayed(d, t_gate, X, y, task, seed=args.seed)
        rows.append(row)
        print(f"  delayed d={d:g} ({time.time()-ts:5.1f}s): T_gate={t_gate}  "
              f"loss={row['final_loss']:.4f}  λ_late={row['late_mean'] if row['late_mean']==row['late_mean'] else float('nan'):.2f}"
              f"{'  DIVERGED' if row['diverged'] or row['late_n_nan'] else ''}",
              flush=True)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== per-seed summary ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    Path("figures").mkdir(parents=True, exist_ok=True)
    out.to_csv(f"figures/delayed_fixedd_s{args.seed}.csv", index=False)
    print(f"\n=> figures/delayed_fixedd_s{args.seed}.csv  (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
