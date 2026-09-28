"""Trajectory figure for Mechanism C: sgd vs wn_proj(budget=1.01) — visualize
the 'weak control authority' result (λ_max peak cut ~24%, but still reaches EoS)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import make_dataset, loss_fn
from steos.models import build_model
from steos.optim import SGD, WNSharpSGD, register_param_names
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(mode, lr=0.05, steps=3000, meas=50, seed=0, budget=1.01):
    torch.manual_seed(seed)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)
    model = build_model("mlp", in_dim=20, hidden_dims=[256, 256], out_dim=1, activation="tanh").to(DEV)
    loss = loss_fn(task); lhw = model.last_hidden_weight()
    opt = (SGD(model.parameters(), lr=lr) if mode == "sgd"
           else WNSharpSGD(model.parameters(), lr=lr, mode="proj", proxy_params=[lhw], budget_factor=budget))
    register_param_names(opt, model)
    steps_r, loss_r, lam_r = [], [], []
    for step in range(steps):
        opt.zero_grad(set_to_none=True); l = loss(model(X), y); l.backward(); opt.step()
        if step % meas == 0:
            lam = float(max_eigenvalue(model, loss, X, y, n_iter=80))
            steps_r.append(step); loss_r.append(float(l.item())); lam_r.append(lam)
    return steps_r, loss_r, lam_r


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
two_over_eta = 2.0 / 0.05
for mode, col, lab in [("sgd", "tab:blue", "SGD"), ("proj", "tab:red", "WN-proj (budget=1.01)")]:
    s, lf, lm = run(mode)
    axes[0].plot(s, lf, "o-", ms=3, lw=0.8, color=col, label=lab)
    axes[1].plot(s, lm, "o-", ms=3, lw=0.8, color=col, label=lab)
axes[1].axhline(two_over_eta, color="0.5", ls="--", lw=1, label=f"2/η={two_over_eta:g}")
axes[0].set_ylabel("loss"); axes[0].set_yscale("log"); axes[0].set_title("loss")
axes[1].set_ylabel("λ_max"); axes[1].set_yscale("log"); axes[1].set_title("sharpness — WN-proj cuts the PEAK ~24% but still reaches EoS")
for ax in axes:
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Mechanism C — weight-norm proxy has weak control authority over sharpness", y=1.02)
fig.tight_layout()
out = Path("figures") / "15_gapC_control_authority.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"saved {out}")
