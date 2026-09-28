"""Mechanism-C falsifiable test: does the free ‖W_lasthidden‖ proxy drive a
useful optimizer? Compare SGD / AdamW / SAM / WN-proj / WN-lr on the MLP EoS
testbed, measuring loss + ‖W_lh‖ + λ_max trajectories.

Two experiments:
  (1) η = 0.05 (stable regime): does C-Proj keep λ_max below the 2/η threshold?
  (2) η = 0.2  (plain SGD DIVERGES, Finding 7): does C-Proj stabilize training?

Usage:
    python scripts/analysis/compare_optimizers.py
Outputs figures/13_optimizer_compare_lr0.05.png, ..._lr0.2.png + a summary CSV.
"""

from __future__ import annotations

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
from steos.optim import SGD, AdamW, WNSharpSGD, register_param_names, sam_step
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OPTIMIZERS = ["sgd", "adamw", "sam", "wn_proj", "wn_lr"]
COLORS = dict(sgd="tab:blue", adamw="tab:orange", sam="tab:green",
              wn_proj="tab:red", wn_lr="tab:purple")


def build_opt(name, model, lr):
    lhw = model.last_hidden_weight()
    if name == "sgd":
        opt = SGD(model.parameters(), lr=lr)
    elif name == "adamw":
        opt = AdamW(model.parameters(), lr=lr)
    elif name == "sam":
        opt = SGD(model.parameters(), lr=lr)  # SAM's base optimizer (plain GD)
    elif name == "wn_proj":
        opt = WNSharpSGD(model.parameters(), lr=lr, mode="proj",
                         proxy_params=[lhw], budget_factor=2.0)
    elif name == "wn_lr":
        opt = WNSharpSGD(model.parameters(), lr=lr, mode="lr",
                         proxy_params=[lhw], lr_beta=1.0, lr_ema=0.99)
    else:
        raise ValueError(name)
    register_param_names(opt, model)
    return opt


def run_one(opt_name, X, y, task, lr, steps=3000, seed=0, measure_every=100, n_iter=100):
    torch.manual_seed(seed)
    model = build_model("mlp", in_dim=task["in_dim"], hidden_dims=[256, 256],
                        out_dim=task["out_dim"], activation="tanh").to(DEV)
    loss = loss_fn(task)
    lhw = model.last_hidden_weight()
    opt = build_opt(opt_name, model, lr)
    rows = []
    for step in range(steps):
        if opt_name == "sam":
            l = sam_step(model, opt, loss, X, y, rho=0.05)
        else:
            opt.zero_grad(set_to_none=True)
            l = float(loss(model(X), y).item())
            loss(model(X), y).backward()
            opt.step()
        with torch.no_grad():
            wn = float(lhw.detach().pow(2).sum().sqrt().item())
        if step % measure_every == 0:
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=n_iter))
        else:
            lam = float("nan")
        rows.append((step, l, wn, lam))
    return pd.DataFrame(rows, columns=["step", "loss", "wnorm_lh", "lambda_max"])


def experiment(lr, steps, out_path):
    print(f"\n##### experiment: η={lr}, steps={steps} #####")
    torch.manual_seed(0)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)
    results = {}
    summary = []
    for name in OPTIMIZERS:
        print(f"  running {name} ...", flush=True)
        df = run_one(name, X, y, task, lr=lr, steps=steps)
        results[name] = df
        finite = df[np.isfinite(df["lambda_max"])]
        summary.append(dict(
            opt=name, lr=lr,
            final_loss=df["loss"].iloc[-1] if np.isfinite(df["loss"].iloc[-1]) else np.nan,
            min_loss=df["loss"].min() if np.isfinite(df["loss"].min()) else np.nan,
            final_wnorm=df["wnorm_lh"].iloc[-1],
            lambda_max_final=float(finite["lambda_max"].iloc[-1]) if len(finite) else np.nan,
            lambda_max_max=float(finite["lambda_max"].max()) if len(finite) else np.nan,
            diverged=int(not np.isfinite(df["loss"].iloc[-1])),
        ))
    summ = pd.DataFrame(summary)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    two_over_eta = 2.0 / lr
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, df in results.items():
        c = COLORS[name]
        axes[0].plot(df["step"], df["loss"], lw=0.9, color=c, label=name)
        m = df.dropna(subset=["lambda_max"])
        axes[1].plot(m["step"], m["lambda_max"], "o-", ms=3, lw=0.7, color=c, label=name)
        axes[2].plot(df["step"], df["wnorm_lh"], lw=0.9, color=c, label=name)
    axes[0].set_ylabel("loss"); axes[0].set_yscale("log"); axes[0].set_title(f"loss  (η={lr})")
    axes[1].set_ylabel("λ_max"); axes[1].set_yscale("log")
    axes[1].axhline(two_over_eta, color="0.5", ls="--", lw=1, label=f"2/η={two_over_eta:g}")
    axes[1].set_title("sharpness (Hessian λ_max)")
    axes[2].set_ylabel("‖W_lasthidden‖"); axes[2].set_title("free sharpness proxy")
    for ax in axes:
        ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    fig.suptitle("Mechanism C — free weight-norm proxy as optimizer control signal", y=1.02)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return summ


def main():
    out = Path("figures")
    s1 = experiment(0.05, 3000, out / "13_optimizer_compare_lr0.05.png")
    s2 = experiment(0.2, 1500, out / "14_optimizer_compare_lr0.2_divergence.png")
    pd.concat([s1, s2]).to_csv(out / "optimizer_compare_summary.csv", index=False)
    print("\n=> figures/13,14 + optimizer_compare_summary.csv")


if __name__ == "__main__":
    main()
