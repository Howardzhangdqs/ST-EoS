"""Multi-sharp migration test — does "spectral migration" manifest when the EoS
landscape has MULTIPLE comparably-sharp directions (λ_1 ≈ λ_2 ≈ 2/η)?

CONTEXT. The rank-k sweep on the standard (1-D-sharp) teacher-student found
λ_max SHARP-ESCAPES to ~265 under rank-1 damping: that landscape has λ_1≈269,
λ_2≈0.6, so there is nowhere for sharpness to migrate to and self-gating simply
removes the EoS barrier along v_1 (the cap is gone, λ_1 grows unboundedly).
The open §6 question: on a MULTI-SHARP landscape (λ_1≈λ_2≈2/η), does damping
v_1 cause λ_2 to RISE toward 2/η (migration signature), and does k>1 matter
NOW (unlike the 1-D k-invariance)?

SETUP (chosen by scripts/experiments/multisharp_screen.py — passes the multi-sharp criterion
λ_2 ≥ 0.2·(2/η) & λ_1/λ_2 < 3). Multi-output isotropic teacher-student: tanh MLP
[256,256] with k=4 outputs whose output-weight columns have equal norm, MSE loss
over the 4 outputs, isotropic Gaussian inputs. At EoS its top Hessian spectrum is
λ_1≈42.8, λ_2≈40.2 (ratio 1.07), λ_3≈20.4, λ_4≈9.6 — two directions pinned at
2/η simultaneously. (Li et al. NeurIPS 2022: the output layer dominates the top
Hessian eigenvalues; k isotropic outputs ⇒ k comparable eigenvalues.)

OPTIMIZERS: sgd (must pin at 2/η), sam (rho=0.05, flat-escape reference),
rankk_k1 (the rank-1 damping test), rankk_k2 / rankk_k5 (does k>1 matter here?).
At each MEASURE_EVERY step the top-12 spectrum is recorded for sgd AND rankk_k1
(the migration diagnostic — contrast λ_2 evolution under damping vs plain EoS).

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/multisharp_experiment.py --seed $s; done

Outputs:
    figures/multisharp_summary_s{seed}.csv    per-opt: opt, final_loss, min_loss,
                                              lambda_max_final, lambda_max_max
    figures/multisharp_spectrum_s{seed}.csv   long: opt, step, idx, lambda_i
                                              (top-12 for sgd & rankk_k1 only)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

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

# --- chosen multi-sharp setup (scripts/experiments/multisharp_screen.py → k4_h256 PASS) ---
K_OUT = 4                  # number of (isotropic) outputs
TEACHER_HIDDEN = [256, 256]
OUT_NORM = 1.0
DIM = 20
N_SAMPLES = 1024
DIFFICULTY = 0.3
DATA_SEED = 1              # fixed — --seed parametrizes MODEL-INIT only

LR = 0.05
STEPS = 2500
MEASURE_EVERY = 100
N_ITER = 100              # power-iteration sweeps for the ground-truth λ_max
SPEC_K = 12              # top-12 eigenvalues for the migration spectrum
SPEC_ITER = 40          # block-power sweeps per spectrum measurement (flat-MGS ⇒ fast)
TWO_OVER_ETA = 2.0 / LR  # = 40
RANKK_KS = [1, 2, 5]
RHO = 0.05               # SAM perturbation radius (matches eos_precond_experiment.py)

# spectrum is recorded only for SGD and rankk_k1 — the migration diagnostic
# (does λ_2 rise as v_1 is damped?). Skipping it for SAM / rankk_k2 / rankk_k5
# halves their wall-clock; λ_max is still measured for all five.
SPEC_OPTS = {"sgd": True, "sam": False, "rankk_k1": True,
             "rankk_k2": False, "rankk_k5": False}


def _new_model(task, seed: int = 0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def _measure(loss, model, X, y, opt_name: str):
    """Ground-truth λ_max (power iter, tol-stopped) always; + top-12 spectrum
    only for SGD / rankk_k1 (the migration diagnostic)."""
    lam_true = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
    specs: Optional[List[float]] = None
    if SPEC_OPTS.get(opt_name, False):
        r = top_k_eigenvalues(model, loss, X, y, k=SPEC_K, n_iter=SPEC_ITER)
        specs = r if isinstance(r, list) else None
    return lam_true, specs


def run(opt_name, X, y, task, seed: int = 0):
    """Run one optimizer on the multi-sharp setup.

    Returns:
        summary: dict(opt, final_loss, min_loss, lambda_max_final, lambda_max_max)
        spectrum_rows: list of (opt, step, idx, lambda_i) at each measurement step
                       (populated for SGD / rankk_k1 only).
    """
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    losses = []
    lam_true_vals = []
    spectrum_rows = []

    def _record_spectrum(step):
        # factored out — identical for every optimizer branch
        if specs is not None:
            for idx, li in enumerate(specs):
                spectrum_rows.append((opt_name, step, idx, float(li)))

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
                _record_spectrum(step)
    elif opt_name == "sam":
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        for step in range(STEPS):
            l = sam_step(model, base, loss, X, y, rho=RHO)
            losses.append(l)
            if step % MEASURE_EVERY == 0:
                lam_true, specs = _measure(loss, model, X, y, opt_name)
                lam_true_vals.append(lam_true)
                _record_spectrum(step)
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
                _record_spectrum(step)
    else:
        raise ValueError(opt_name)

    summary = dict(
        opt=opt_name,
        final_loss=float(losses[-1]) if losses else float("nan"),
        min_loss=float(min(losses)) if losses else float("nan"),
        lambda_max_final=float(lam_true_vals[-1]) if lam_true_vals else float("nan"),
        lambda_max_max=float(max(lam_true_vals)) if lam_true_vals else float("nan"),
    )
    return summary, spectrum_rows


def main():
    global MEASURE_EVERY
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-every", type=int, default=MEASURE_EVERY,
                    help="measurement period in steps (default 100; the "
                         "fine-grained fig:multisharp rerun uses 10)")
    args = ap.parse_args()
    MEASURE_EVERY = args.measure_every
    t0 = time.time()
    print(f"\n##### Multi-sharp migration test: k_out={K_OUT}, hidden={TEACHER_HIDDEN}, "
          f"η={LR}, steps={STEPS}, 2/η={TWO_OVER_ETA:g}, DEV={DEV}, seed={args.seed} #####")
    # data seed fixed → every model-init seed sees identical data (matches
    # rankk_sweep_experiment.py convention; --seed parametrizes init only)
    torch.manual_seed(DATA_SEED)
    X, y, task = make_dataset(
        "multisharp_teacher_student", n=N_SAMPLES, dim=DIM, difficulty=DIFFICULTY,
        teacher_hidden=TEACHER_HIDDEN, out_dim=K_OUT, out_norm=OUT_NORM,
        device=DEV, seed=DATA_SEED,
    )

    opt_names = ["sgd", "sam"] + [f"rankk_k{k}" for k in RANKK_KS]
    summary_rows, all_spectrum = [], []
    for name in opt_names:
        ts = time.time()
        summ, spec = run(name, X, y, task, seed=args.seed)
        summary_rows.append(summ)
        all_spectrum.extend(spec)
        print(f"  {name:10s} ({time.time()-ts:5.1f}s): loss={summ['final_loss']:.4f}  "
              f"λ_max_final={summ['lambda_max_final']:7.2f}  "
              f"λ_max_peak={summ['lambda_max_max']:.1f}", flush=True)

    summ = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== summary ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- verdict ----------------------------------------------------------
    lam_thresh = 0.7 * TWO_OVER_ETA
    print(f"\n=== VERDICT (escape-to-flat needs λ_max_final < {lam_thresh:.1f} = 0.7·2/η; "
          f"sharp-escape >> 2/η = {TWO_OVER_ETA:g}) ===")
    for k in RANKK_KS:
        name = f"rankk_k{k}"
        row = summ[summ.opt == name].iloc[0]
        if row["lambda_max_final"] < lam_thresh:
            cls = "FLAT-ESCAPE"
        elif row["lambda_max_final"] < 1.4 * TWO_OVER_ETA:
            cls = "lock@2/η"
        else:
            cls = "SHARP-ESCAPE (>>2/η)"
        print(f"  {name:10s}: λ_max_final={row['lambda_max_final']:7.2f}  → {cls}")
    for ref in ("sgd", "sam"):
        row = summ[summ.opt == ref].iloc[0]
        print(f"  {ref:8s}: λ_max_final={row['lambda_max_final']:7.2f}")

    Path("figures").mkdir(parents=True, exist_ok=True)
    summ.to_csv(f"figures/multisharp_summary_s{args.seed}.csv", index=False)
    pd.DataFrame(all_spectrum, columns=["opt", "step", "idx", "lambda_i"]).to_csv(
        f"figures/multisharp_spectrum_s{args.seed}.csv", index=False)
    print(f"\n=> figures/multisharp_summary_s{args.seed}.csv "
          f"+ figures/multisharp_spectrum_s{args.seed}.csv  (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
