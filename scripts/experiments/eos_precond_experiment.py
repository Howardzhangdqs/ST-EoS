"""EoS-Precond falsification experiment.

Design 1 from the optimizer-design analysis: rank-1 curvature preconditioning
along the FREE top eigenvector (v̂ = unit Δθ_{t−1}, enabled by Finding 4's
|cos(Δθ, v_top)| → 0.998 at EoS), with one HVP/step for the Rayleigh quotient
λ̂ = v̂ᵀH v̂. See steos/optim/eos_precond.py for the full mechanism.

The control-theoretic prediction (the project's negative result, sharpened):
controller actions (W-cap / γ-cap / η-scaling) all collapse back to λ_max ≈ 2/η
because the EoS plant has DC gain −1. Only PLANT actions escape — SAM does
(F12: λ_max_final = 21 < 40 = 2/η). EoS-Precond is a plant action, so the
question is whether it escapes too.

PASS CRITERION (decisive):  λ_max_final(eos_precond) < 0.7·(2/η) = 28
                            AND  final_loss ≤ 1.05·SGD_final_loss
If it hits 2/η like every WN variant → the cheap-plant-action cell is also
closed, strengthening the negative result.

Setup is IDENTICAL to compare_optimizers.py / gapC_retry.py (same dataset seed,
model, η, steps) so the SGD baseline must reproduce λ_max_final ≈ 39.8.

Usage:
    python scripts/experiments/eos_precond_experiment.py
Outputs: figures/22_eos_precond_falsification.png + figures/eos_precond_summary.csv
"""
from __future__ import annotations
import argparse  # multi-seed CIs: --seed parametrization
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.optim import SGD, AdamW, sam_step, EoSPrecondState, eos_precond_step, register_param_names
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
STEPS = 3000
MEASURE_EVERY = 100
N_ITER = 100
TWO_OVER_ETA = 2.0 / LR  # = 40

COLORS = dict(sgd="tab:blue", sam="tab:green", adamw="tab:orange",
              eos_precond="tab:red", eos_precond_reacq="tab:purple")


def _new_model(task, seed=0):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def run(opt_name, X, y, task, seed=0):
    """Run one optimizer; return a per-step DataFrame with loss + measured λ_max
    (+ EoS-Precond diagnostics lam_hat / cos_g_vhat / divisor when applicable)."""
    model = _new_model(task, seed=seed)
    loss = loss_fn(task)
    rows = []
    if opt_name in ("eos_precond", "eos_precond_reacq"):
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        st = EoSPrecondState()
        reacq = 0.5 if opt_name == "eos_precond_reacq" else None
        for step in range(STEPS):
            l = eos_precond_step(model, base, loss, X, y, st, lr=LR, reacquire_threshold=reacq)
            m = st.metrics()
            if step % MEASURE_EVERY == 0:
                lam_true = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER))
            else:
                lam_true = float("nan")
            rows.append((step, l, lam_true, m["lam_hat"], m["cos_g_vhat"], m["divisor"], m["n_reacquired"]))
        cols = ["step", "loss", "lambda_max", "lam_hat", "cos_g_vhat", "divisor", "n_reacquired"]
    elif opt_name == "sam":
        base = SGD(model.parameters(), lr=LR)
        register_param_names(base, model)
        for step in range(STEPS):
            l = sam_step(model, base, loss, X, y, rho=0.05)
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER)) if step % MEASURE_EVERY == 0 else float("nan")
            rows.append((step, l, lam))
        cols = ["step", "loss", "lambda_max"]
    elif opt_name == "adamw":
        opt = AdamW(model.parameters(), lr=LR)
        register_param_names(opt, model)
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True); l = float(loss(model(X), y).item()); loss(model(X), y).backward(); opt.step()
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER)) if step % MEASURE_EVERY == 0 else float("nan")
            rows.append((step, l, lam))
        cols = ["step", "loss", "lambda_max"]
    elif opt_name == "sgd":
        opt = SGD(model.parameters(), lr=LR)
        register_param_names(opt, model)
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True); l = float(loss(model(X), y).item()); loss(model(X), y).backward(); opt.step()
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=N_ITER)) if step % MEASURE_EVERY == 0 else float("nan")
            rows.append((step, l, lam))
        cols = ["step", "loss", "lambda_max"]
    else:
        raise ValueError(opt_name)
    return pd.DataFrame(rows, columns=cols)


def main():
    # multi-seed CIs: --seed parametrizes MODEL-INIT only; data seed stays fixed (seed=1) below
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"\n##### EoS-Precond falsification: η={LR}, steps={STEPS}, 2/η={TWO_OVER_ETA:g} #####")
    torch.manual_seed(1)  # data seed — kept fixed so all seeds see identical data
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)

    results, summary = {}, []
    for name in ["sgd", "sam", "adamw", "eos_precond", "eos_precond_reacq"]:
        print(f"  running {name} ...", flush=True)
        df = run(name, X, y, task, seed=args.seed)  # multi-seed CIs: model-init seed from --seed
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

    # ---- verdict ----------------------------------------------------------
    sgd_loss = float(summ.loc[summ.opt == "sgd", "final_loss"].iloc[0])
    lam_thresh = 0.7 * TWO_OVER_ETA
    print(f"\n=== FALSIFICATION VERDICT (Design 1: EoS-Precond) — pass needs λ_max_final<{lam_thresh:.1f}=0.7·2/η AND loss≤1.05·SGD={1.05*sgd_loss:.4f} ===")
    for variant in ["eos_precond", "eos_precond_reacq"]:
        if variant not in summ["opt"].values:
            continue
        row = summ[summ.opt == variant].iloc[0]
        sp = row["lambda_max_final"] < lam_thresh
        lp = row["final_loss"] <= 1.05 * sgd_loss
        nreq = int(results[variant]["n_reacquired"].iloc[-1]) if "n_reacquired" in results[variant] else 0
        tag = "FREE v̂" if variant == "eos_precond" else f"RE-ACQ v_top ({nreq}×)"
        print(f"  {variant:20s} [{tag}]: λ_max_final={row['lambda_max_final']:.2f} {'PASS' if sp else 'FAIL':>4} | "
              f"loss={row['final_loss']:.4f} {'PASS' if lp else 'FAIL':>4} | peak={row['lambda_max_max']:.1f}")
    ep_lam = float(summ.loc[summ.opt == "eos_precond", "lambda_max_final"].iloc[0])
    ep_loss = float(summ.loc[summ.opt == "eos_precond", "final_loss"].iloc[0])
    has_reacq = "eos_precond_reacq" in summ["opt"].values
    ra_lam = float(summ.loc[summ.opt == "eos_precond_reacq", "lambda_max_final"].iloc[0]) if has_reacq else ep_lam
    if has_reacq:
        print(f"\n  Migration-vs-staleness test: FREE v̂ λ_max_final={ep_lam:.1f}  vs  RE-ACQ λ_max_final={ra_lam:.1f}  (2/η={TWO_OVER_ETA:g})")
        if ra_lam >= lam_thresh and ep_lam >= lam_thresh:
            print("  => BOTH variants collapse to 2/η: the failure is STRUCTURAL (spectral migration),")
            print("     not a v̂-staleness artifact. Even a *correct* v_top cannot let rank-1 escape EoS.")
            print("     This airtights the negative result: rank-1 plant-action preconditioning fails too.")
        elif ra_lam < lam_thresh:
            print("  => RE-ACQ escapes but FREE v̂ does not: failure was a tracking artifact — revivable.")
    if ep_lam >= lam_thresh and (not has_reacq or ra_lam >= lam_thresh):
        print("  => Contrast: SAM (global dual-ascent over ALL dirs) escapes to ~21; rank-1 (one dir) cannot.")
        print("     Sharpness is a near-conserved budget that migrates to any un-dampened eigenvector.")

    summ.to_csv(f"figures/eos_precond_summary_s{args.seed}.csv", index=False)  # multi-seed CIs: seed-suffixed path

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    # (a) loss
    for name, df in results.items():
        axes[0, 0].plot(df["step"], df["loss"], lw=0.9, color=COLORS[name], label=name)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("loss"); axes[0, 0].set_title("loss")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(fontsize=8)
    # (b) measured λ_max — THE falsification panel
    for name, df in results.items():
        m = df.dropna(subset=["lambda_max"])
        axes[0, 1].plot(m["step"], m["lambda_max"], "o-", ms=3, lw=0.7, color=COLORS[name], label=name)
    axes[0, 1].axhline(TWO_OVER_ETA, color="0.4", ls="--", lw=1.2, label=f"2/η={TWO_OVER_ETA:g}")
    axes[0, 1].axhline(lam_thresh, color="tab:red", ls=":", lw=1.2, label=f"pass<{lam_thresh:g}")
    axes[0, 1].set_yscale("log"); axes[0, 1].set_ylabel("λ_max (power iter, ground truth)")
    axes[0, 1].set_title("sharpness — does EoS-Precond escape 2/η?"); axes[0, 1].grid(alpha=0.3); axes[0, 1].legend(fontsize=7)
    # (c) EoS-Precond diagnostics: cheap λ̂ (per-step, free) vs measured λ_max
    df_ep = results["eos_precond"]
    m = df_ep.dropna(subset=["lambda_max"])
    axes[1, 0].plot(df_ep["step"], df_ep["lam_hat"], lw=0.6, color="tab:red", alpha=0.7, label="λ̂ = v̂ᵀH v̂ (1 HVP, free v̂)")
    axes[1, 0].plot(m["step"], m["lambda_max"], "ko", ms=3, label="λ_max (power iter)")
    axes[1, 0].axhline(TWO_OVER_ETA, color="0.4", ls="--", lw=1)
    axes[1, 0].set_yscale("log"); axes[1, 0].set_ylabel("sharpness"); axes[1, 0].set_xlabel("step")
    axes[1, 0].set_title("EoS-Precond: does free λ̂ track λ_max? (F4 check)"); axes[1, 0].grid(alpha=0.3); axes[1, 0].legend(fontsize=7)
    # (d) alignment + divisor — does the mechanism engage at EoS?
    ax2 = axes[1, 1]
    ax2.plot(df_ep["step"], df_ep["cos_g_vhat"], lw=0.7, color="tab:purple", label="|cos(g, v̂)| (F4 alignment)")
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("|cos(g, v̂)|", color="tab:purple"); ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax2.set_xlabel("step"); ax2.set_title("EoS-Precond: alignment + divisor (does it fire?)"); ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    fired = (df_ep["divisor"] > 1.0 + 1e-9).astype(float)
    ax3.scatter(df_ep["step"], df_ep["divisor"], s=3, c=fired.map({0.0: "0.7", 1.0: "tab:red"}), alpha=0.5, label="divisor (red=fired)")
    ax3.set_ylabel("divisor = max(λ̂/(2/η), 1)", color="tab:red"); ax3.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")

    fig.suptitle(f"EoS-Precond falsification — η={LR}, 2/η={TWO_OVER_ETA:g}\n"
                 f"FREE v̂: λ_max_final={ep_lam:.1f} ({'PASS' if ep_lam < lam_thresh else 'FAIL'}<{lam_thresh:g}); "
                 f"RE-ACQ: λ_max_final={ra_lam:.1f} ({'PASS' if ra_lam < lam_thresh else 'FAIL'})", y=1.01)
    out = Path(f"figures/22_eos_precond_falsification_s{args.seed}.png")  # multi-seed CIs: seed-suffixed path
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\n=> {out} + figures/eos_precond_summary_s{args.seed}.csv")


if __name__ == "__main__":
    main()
