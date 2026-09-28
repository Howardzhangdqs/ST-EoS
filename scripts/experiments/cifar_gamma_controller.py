"""Mechanism C′ payoff test: does capping ‖BN γ‖ (the Finding-14 sharpness
proxy) have ACTUATOR authority under BatchNorm — unlike the W-cap (Mechanism C,
which was a no-op / wrong-signed on BN nets)?

Compare on CIFAR-10 PreAct-ResNet (BN), μ=0.9 SGD:
  - baseline: plain SGD
  - gamma_cap: SGD + project all BN γ onto budget = 1.02·‖γ_init‖ each step
Record λ_max trajectory + accuracy. If gamma_cap cuts the λ_max peak without
hurting accuracy → γ has real actuator authority → the optimizer angle revives.
"""
from __future__ import annotations
import argparse, sys  # multi-seed CIs: --seed parametrization (argparse)
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data.cifar import load_cifar10
from steos.models import build_model
from steos.optim import SGD, WNSharpSGD, register_param_names
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR, MOM, BS, STEPS = 0.1, 0.9, 128, 1500
SHARP_EVERY, SHARP_BATCH, NITER = 50, 256, 60


def bn_gammas(model):
    return [m.weight for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def acc_on(model, X, y, bs=1000):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        c += int((model(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum())
    model.train(); return c / len(X)


def run(mode, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, _, Xte, yte = load_cifar10(device=DEV)
    N = len(Xtr)
    idx_e = np.random.default_rng(123).integers(0, N, SHARP_BATCH)
    Xe, ye = Xtr[idx_e], ytr[idx_e]
    model = build_model("resnet", widths=(16, 32, 64), blocks=(2, 2, 2), num_classes=10).to(DEV)
    if mode == "baseline":
        opt = SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=1e-4)
    else:  # gamma_cap
        opt = WNSharpSGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=1e-4,
                         mode="proj", proxy_params=bn_gammas(model), budget_factor=1.02)
    register_param_names(opt, model)
    rng = np.random.default_rng(seed); rows = []; n_triggered = 0
    for step in range(STEPS):
        idx = rng.integers(0, N, BS); Xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True); F.cross_entropy(model(Xb), yb).backward(); opt.step()
        if step % SHARP_EVERY == 0 or step == STEPS - 1:
            model.eval()
            try:
                lam = float(max_eigenvalue(model, lambda p, y: F.cross_entropy(p, y), Xe, ye, n_iter=NITER))
            finally:
                model.train()
            rows.append((step, lam, acc_on(model, Xte[:2000], yte[:2000])))
            if hasattr(opt, "_n_triggered"):
                pass
    df = pd.DataFrame(rows, columns=["step", "lambda_max", "acc"])
    return df


def main():
    # multi-seed CIs: --seed parametrizes torch/numpy init (data + sharpness-subset RNG stay fixed)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("baseline SGD ...", flush=True); base = run("baseline", seed=args.seed)  # multi-seed CIs: init seed from --seed
    print("gamma_cap SGD ...", flush=True); gcap = run("gamma_cap", seed=args.seed)
    print("\n=== λ_max peak / final ===")
    print(f"  baseline : peak {base['lambda_max'].max():.1f}  final {base['lambda_max'].iloc[-1]:.1f}  acc {base['acc'].iloc[-1]:.3f}")
    print(f"  gamma_cap: peak {gcap['lambda_max'].max():.1f}  final {gcap['lambda_max'].iloc[-1]:.1f}  acc {gcap['acc'].iloc[-1]:.3f}")
    peak_cut = (1 - gcap['lambda_max'].max() / base['lambda_max'].max()) * 100
    print(f"  => γ-cap cuts λ_max peak by {peak_cut:.1f}%  "
          f"({'actuator authority YES' if peak_cut > 10 else 'weak/no actuator authority'})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(base["step"], base["lambda_max"], "o-", ms=3, color="tab:blue", label="baseline SGD")
    ax[0].plot(gcap["step"], gcap["lambda_max"], "s-", ms=3, color="tab:red", label="SGD + γ-cap")
    ax[0].axhline(2/LR, color="0.6", ls="--", lw=1, label=f"2/η={2/LR:g}")
    ax[0].set_ylabel("λ_max"); ax[0].set_yscale("log"); ax[0].set_xlabel("step")
    ax[0].set_title("sharpness"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(base["step"], base["acc"], "o-", ms=3, color="tab:blue", label="baseline")
    ax[1].plot(gcap["step"], gcap["acc"], "s-", ms=3, color="tab:red", label="γ-cap")
    ax[1].set_ylabel("test acc"); ax[1].set_xlabel("step"); ax[1].set_title("accuracy")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Mechanism C′ — does capping ‖BN γ‖ control sharpness under BatchNorm?", y=1.02)
    fig.tight_layout(); Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/18_gamma_controller_s{args.seed}.png", dpi=140, bbox_inches="tight"); plt.close(fig)  # multi-seed CIs: seed-suffixed path
    pd.concat([base.assign(opt="baseline"), gcap.assign(opt="gamma_cap")]).to_csv(
        f"figures/gamma_controller_s{args.seed}.csv", index=False)  # multi-seed CIs: seed-suffixed path
    print(f"saved figures/18_gamma_controller_s{args.seed}.png")


if __name__ == "__main__":
    main()
