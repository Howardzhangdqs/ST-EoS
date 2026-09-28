"""SAM (Sharpness-Aware Minimization, Foret et al. 2021) as a two-step wrapper.

SAM is a baseline against the novel WNSharpSGD: it also manipulates the
sharpness, but via an explicit perturbation + recompute (costs ~2×
forward-backward) rather than a free proxy. Implemented as a function used
inside a manual training loop (it does not fit the single-step optimizer API)."""

from __future__ import annotations

import torch


@torch.no_grad()
def _global_grad_norm(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum().item())
    return s ** 0.5


def sam_step(model, base_optimizer, loss_fn, X, y, rho: float = 0.05):
    """One SAM step: perturb uphill by ρ·g/‖g‖, recompute gradient there, then
    descend from the ORIGINAL params using that gradient."""
    # 1. gradient at θ
    base_optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    gnorm = _global_grad_norm(model)
    if gnorm < 1e-12:
        return float(loss.item())

    # 2. remember θ, perturb to θ + ρ g/‖g‖
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    scale = rho / (gnorm + 1e-12)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad * scale)  # ASCEND (gradient ascent perturbation)

    # 3. gradient at θ_pert
    base_optimizer.zero_grad(set_to_none=True)
    loss_pert = loss_fn(model(X), y)
    loss_pert.backward()

    # 4. restore θ (keep the perturbed-point gradient), then descend
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(orig[n])
    base_optimizer.step()
    return float(loss.item())
