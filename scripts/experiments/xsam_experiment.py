"""X-SAM ablation experiment — tests the §5 falsifiable prediction that X-SAM
(Duan et al. 2026, arXiv:2601.10251) escapes the EoS threshold 2/η ONLY because
of its GLOBAL SAM perturbation component; its rank-1 top-eigenvector gradient
correction contributes nothing to escape (same spectral-migration failure as
our rank-1 EoS-Precond). See steos/optim/xsam.py for the full mechanism.

The 3-variant ablation isolates which component drives escape:
  1. sam             — vanilla SAM (reference). Predict: escapes, λ_max_final≈21.
  2. xsam_full       — SAM perturbation + rank-1 correction. Predict: escapes
                       ≈ like SAM (rank-1 adds nothing).
  3. xsam_rank1only  — rank-1 correction, NO SAM. Predict: locks at 2/η≈40
                       (this is essentially our EoS-Precond).
(sgd is included as the 2/η-locking reference.)

Falsification logic: if xsam_full ≈ sam (overlapping λ_max_final CIs) AND
xsam_rank1only ≈ 2/η  ⇒  escape is ENTIRELY the global SAM component, §5
prediction confirmed. If xsam_full escapes MORE than sam (disjoint CIs) ⇒ the
rank-1 correction contributes, prediction weakened.

Setup is IDENTICAL to eos_precond_experiment.py / compare_optimizers.py (same
dataset seed=1 fixed, model, η=0.05, 3000 steps) so the SGD baseline must
reproduce λ_max_final ≈ 39.8 and SAM ≈ 21.4.

Usage:
    for s in 0 1 2 3 4; do python scripts/experiments/xsam_experiment.py --seed $s; done
Outputs: figures/xsam_summary_s{seed}.csv  (columns: opt, final_loss, min_loss,
         lambda_max_final, lambda_max_max).
"""
from __future__ import annotations
import argparse  # multi-seed CIs: --seed parametrizes MODEL-INIT only
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.optim import (SGD, sam_step, xsam_step, XSAMState,
                         register_param_names)
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100
TWO_OVER_ETA = 2.0 / LR  # = 40


def _new_model(task, seed=0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def run(opt_name, X, y, task, seed=0):
    """Run one optimizer; return a per-step DataFrame with loss + measured λ_max."""
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    rows = []
    if opt_name in ("xsam_full", "xsam_rank1only"):
        use_sam = (opt_name == "xsam_full")
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = XSAMState()
        for step in range(STEPS):
            l = xsam_step(model, base, loss, X, y, st, lr=LR, rho=0.05,
                          use_sam=use_sam)
            if step % MEASURE_EVERY == 0:
                lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
            else:
                lam = float("nan")
            rows.append((step, l, lam))
        cols = ["step", "loss", "lambda_max"]
    elif opt_name == "sam":
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        for step in range(STEPS):
            l = sam_step(model, base, loss, X, y, rho=0.05)
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER)) if step % MEASURE_EVERY == 0 else float("nan")
            rows.append((step, l, lam))
        cols = ["step", "loss", "lambda_max"]
    elif opt_name == "sgd":
        opt = SGD(model.parameters(), lr=LR)
        register_param_names(opt, model)
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True)
            l = float(loss(model(X), y).item())
            loss(model(X), y).backward()
            opt.step()
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER)) if step % MEASURE_EVERY == 0 else float("nan")
            rows.append((step, l, lam))
        cols = ["step", "loss", "lambda_max"]
    else:
        raise ValueError(opt_name)
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"\n##### X-SAM ablation: η={LR}, steps={STEPS}, 2/η={TWO_OVER_ETA:g} #####")
    torch.manual_seed(1)  # data seed — kept fixed so all seeds see identical data
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)

    results, summary = {}, []
    for name in ["sgd", "sam", "xsam_full", "xsam_rank1only"]:
        print(f"  running {name} ...", flush=True)
        df = run(name, X, y, task, seed=args.seed)  # model-init seed from --seed
        results[name] = df
        finite = df[np.isfinite(df["lambda_max"])]
        summary.append(dict(
            opt=name,
            final_loss=float(df["loss"].iloc[-1]) if np.isfinite(df["loss"].iloc[-1]) else np.nan,
            min_loss=float(df["loss"].min()),
            lambda_max_final=float(finite["lambda_max"].iloc[-1]) if len(finite) else np.nan,
            lambda_max_max=float(finite["lambda_max"].max()) if len(finite) else np.nan,
        ))
    summ = pd.DataFrame(summary)
    pd.set_option("display.width", 200)
    print("\n=== summary ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- verdict (single-seed preview; the CI-overlap verdict is in analyze_xsam.py) ----
    def get(opt, col):
        return float(summ.loc[summ.opt == opt, col].iloc[0])
    sam_lam = get("sam", "lambda_max_final")
    xf_lam = get("xsam_full", "lambda_max_final")
    xr_lam = get("xsam_rank1only", "lambda_max_final")
    print(f"\n=== ABLATION VERDICT (seed={args.seed}; multi-seed CIs in analyze_xsam.py) ===")
    print(f"  λ_max_final:  sgd={get('sgd','lambda_max_final'):.2f}  "
          f"sam={sam_lam:.2f}  xsam_full={xf_lam:.2f}  "
          f"xsam_rank1only={xr_lam:.2f}   (2/η={TWO_OVER_ETA:g})")
    print(f"  xsam_full − sam        = {xf_lam - sam_lam:+.2f}  "
          f"({'≈ overlapping (rank-1 adds nothing)' if abs(xf_lam - sam_lam) < 3.0 else 'DIFFERS — check CIs'})")
    print(f"  xsam_rank1only − 2/η   = {xr_lam - TWO_OVER_ETA:+.2f}  "
          f"({'locks at 2/η (rank-1 alone fails)' if abs(xr_lam - TWO_OVER_ETA) < 3.0 else 'does NOT lock — investigate'})")

    summ.to_csv(f"figures/xsam_summary_s{args.seed}.csv", index=False)
    print(f"\n=> figures/xsam_summary_s{args.seed}.csv")


if __name__ == "__main__":
    main()
