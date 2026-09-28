"""Fair aggressive-retry for Mechanism C: does the weight-norm proxy have ANY
control authority over λ_max if we set a TIGHT budget / aggressive LR response?
(Defaults in compare_optimizers.py were too loose — the projection never fired.)

Sweep: wn_proj budget_factor ∈ {1.01,1.02,1.05,1.1}, wn_lr β ∈ {2,4,8}, at η=0.05.
Reports λ_max_max and final loss vs plain SGD. If even tight settings barely move
λ_max, the proxy is confirmed diagnostic-only (no control authority).
"""
from __future__ import annotations
import argparse, sys  # multi-seed CIs: --seed parametrization (argparse)
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import make_dataset, loss_fn
from steos.models import build_model
from steos.optim import SGD, WNSharpSGD, register_param_names
from steos.sharpness import max_eigenvalue

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# multi-seed CIs: --seed parametrizes MODEL-INIT only; data seed stays fixed at 1 inside run()
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
SEED = ap.parse_args().seed


def run(mode, lr=0.05, steps=3000, measure_every=100, seed=0, **opt_kw):
    torch.manual_seed(seed)
    X, y, task = make_dataset("teacher_student", n=1024, dim=20, difficulty=0.3,
                              teacher_hidden=[128, 128], device=DEV, seed=1)
    model = build_model("mlp", in_dim=20, hidden_dims=[256, 256], out_dim=1, activation="tanh").to(DEV)
    loss = loss_fn(task); lhw = model.last_hidden_weight()
    if mode == "sgd":
        opt = SGD(model.parameters(), lr=lr)
    else:
        opt = WNSharpSGD(model.parameters(), lr=lr, mode=mode, proxy_params=[lhw], **opt_kw)
    register_param_names(opt, model)
    lams = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True); loss(model(X), y).backward(); opt.step()
        if step % measure_every == 0:
            lams.append(float(max_eigenvalue(model, loss, X, y, n_iter=80)))
    return dict(final_loss=float(loss(model(X), y).item()),
                lambda_max_max=max(lams), lambda_max_final=lams[-1],
                final_wnorm=float(lhw.detach().pow(2).sum().sqrt().item()),
                n_proj_triggered=getattr(opt, "_n_triggered", None))


# instrument WNSharpSGD to count projection triggers
_orig_step = WNSharpSGD.step
def _counting_step(self, closure=None):
    import torch as _t
    with _t.no_grad():
        before = [float(p.detach().pow(2).sum().sqrt().item()) for p in self.proxy_params]
    r = _orig_step(self, closure)
    with _t.no_grad():
        after = [float(p.detach().pow(2).sum().sqrt().item()) for p in self.proxy_params]
    if self.mode == "proj" and not hasattr(self, "_n_triggered"):
        self._n_triggered = 0
    if self.mode == "proj":
        for i in range(len(before)):
            if after[i] < before[i] - 1e-9:  # projection shrank it
                self._n_triggered = self._n_triggered + 1
                break
    return r
WNSharpSGD.step = _counting_step

rows = []
rows.append({"variant": "sgd", **run("sgd", seed=SEED)})  # multi-seed CIs: model-init seed from --seed
for bf in [1.01, 1.02, 1.05, 1.1, 1.5]:
    rows.append({"variant": f"wn_proj budget={bf}", **run("proj", budget_factor=bf, seed=SEED)})
for beta in [2, 4, 8]:
    rows.append({"variant": f"wn_lr beta={beta}", **run("lr", lr_beta=beta, seed=SEED)})

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== Mechanism C aggressive retry (η=0.05) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nSGD baseline λ_max_max = 68. If all wn_proj/wn_lr rows stay ≥ ~55, the")
print("proxy has no control authority over sharpness → confirmed diagnostic-only.")
df.to_csv(f"figures/gapC_aggressive_retry_s{SEED}.csv", index=False)  # multi-seed CIs: seed-suffixed path
