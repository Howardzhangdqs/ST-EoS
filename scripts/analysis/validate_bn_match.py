"""Round-7 BN-protocol validation (one-step buffer-evolution contract).

The three double-update arms must produce BIT-IDENTICAL BatchNorm
running-stat evolution after one step from the same (θ, batch):

  sam               : fwd@θ (upd 1) → backward → perturb → fwd@θ+ε (upd 2)
  sgd_bn2x          : fwd@θ (upd 1) → backward → perturb → no-grad fwd@θ+ε (upd 2)
  linsam_bnmatched  : fwd@θ (upd 1) → HVP → set corrected grad → perturb by RAW g
                      → no-grad fwd@θ+ε (upd 2)

and sam_bnfreeze must leave the buffers exactly at initialization.

Usage: .venv/bin/python scripts/analysis/validate_bn_match.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Deterministic CUDA (same as the --deterministic flag of the training
# script): without it, non-deterministic conv-backward makes g — and hence
# the perturbation ε — differ across arms at the 1e-6 level, which is a
# measurement artifact, not a semantics difference.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from steos.data.cifar import load_cifar                       # noqa: E402
from steos.optim import SGD                                   # noqa: E402
from steos.optim.sam import sam_step                          # noqa: E402
from steos.optim.linsam import LinSAMState, linsam_bnmatched_step  # noqa: E402
from scripts.train_cifar100_resnet18 import (                 # noqa: E402
    build_resnet18, _bn_buffers, _restore_bn_buffers, _global_grad_norm)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RHO = 0.05


def bn_snapshot(model) -> list[torch.Tensor]:
    return [m.running_mean.detach().clone() for m in model.modules()
            if isinstance(m, torch.nn.BatchNorm2d)]


def max_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def main() -> None:
    torch.manual_seed(0)
    Xtr, ytr, *_ = load_cifar(n_classes=100, device=DEV)
    idx = torch.arange(128, device=DEV)
    Xb, yb = Xtr[idx], ytr[idx]
    ce = lambda p, y: F.cross_entropy(p, y)

    results = {}
    for arm in ["sam", "sgd_bn2x", "linsam_bnmatched", "sam_bnfreeze"]:
        torch.manual_seed(0)
        model = build_resnet18(num_classes=100).to(DEV)
        model.train()
        base = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        init = bn_snapshot(model)

        if arm == "sam":
            sam_step(model, base, ce, Xb, yb, rho=RHO)
        elif arm == "sam_bnfreeze":
            bufs = _bn_buffers(model)
            sam_step(model, base, ce, Xb, yb, rho=RHO)
            _restore_bn_buffers(bufs)
        elif arm == "sgd_bn2x":
            # --- copied verbatim from the sgd_bn2x dispatch branch ---
            base.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xb), yb)
            loss.backward()
            gnorm2 = _global_grad_norm(model)
            if gnorm2 > 1e-12:
                scale2 = RHO / (gnorm2 + 1e-12)
                orig2 = {n: p.detach().clone() for n, p in model.named_parameters()}
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.add_(p.grad * scale2)
                    _ = ce(model(Xb), yb)
                    for n, p in model.named_parameters():
                        p.copy_(orig2[n])
            base.step()
        elif arm == "linsam_bnmatched":
            linsam_bnmatched_step(model, base, ce, Xb, yb, LinSAMState(), rho=RHO)

        results[arm] = bn_snapshot(model)
        print(f"{arm:18s} |Δ| vs init (max over layers): "
              f"{max_diff(results[arm], init):.3e}", flush=True)

    ref = results["sam"]
    print("\n--- contract checks ---")
    for arm in ["sgd_bn2x", "linsam_bnmatched"]:
        d = max_diff(results[arm], ref)
        ok = d == 0.0
        print(f"{arm:18s} vs sam      : max|Δ| = {d:.3e}  "
              f"{'BIT-IDENTICAL ✓' if ok else 'MISMATCH ✗'}")
    d = max_diff(results["sam_bnfreeze"], init)
    print(f"{'sam_bnfreeze':18s} vs init     : max|Δ| = {d:.3e}  "
          f"{'FROZEN ✓' if d == 0.0 else 'NOT FROZEN ✗'}")


if __name__ == "__main__":
    main()
