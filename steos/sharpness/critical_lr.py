"""Critical sharpness λ_c — adapted from facebookresearch/scalable-curvature
(utils/measure_critical_lr.py), single-process (no DDP/all-reduce).

Definition (Kalra et al., arXiv:2601.16979):
    η_c = inf{ η > 0 : L(θ − η Δθ) > L(θ) }      (smallest LR that increases loss)
    λ_c = 2 / η_c
where Δθ is the optimizer's update direction at the current θ. For plain GD on
a local quadratic, λ_c == λ_max exactly, which makes a clean cross-check of the
power-iteration estimate.

Algorithm: exponential bracketing + bisection along Δθ using ONLY forward
passes (a "virtual step" recomputes the update at an arbitrary LR via the
optimizer's `_compute_update` without mutating its state). Typically 5–8
forward passes per measurement.

Requires the custom SGD from `steos.optim` (exposes `_compute_update`) and
`param_names` registered in each param group (see `register_param_names`).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@contextmanager
def atParams(model: nn.Module, update_dict: Dict[str, Tensor]):
    """Temporarily add `update_dict[name]` to each param, restore on exit."""
    cache = {}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if name in update_dict:
                    cache[name] = param.clone()
                    param.add_(update_dict[name])
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in cache:
                    param.data.copy_(cache.pop(name))


@torch.no_grad()
def virtual_step(model: nn.Module, optim: torch.optim.Optimizer, lr: float) -> Dict[str, Tensor]:
    """Compute the Δθ the optimizer WOULD apply at `lr`, without applying it."""
    updates: Dict[str, Tensor] = {}
    for group in optim.param_groups:
        names = group.get("param_names", [n for n, _ in model.named_parameters()])
        for name, param in zip(names, group["params"]):
            if param.grad is None:
                continue
            if not hasattr(optim, "_compute_update"):
                raise AttributeError(
                    "Optimizer must expose `_compute_update`; use steos.optim.SGD."
                )
            state = optim.state[param]
            updates[name] = optim._compute_update(
                param, param.grad, state, group, lr_override=lr, virtual=True
            )
    return updates


def measure_critical_lr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    loss_closure: Callable[[Any], float],
    lr_guess: float = 1e-3,
    tol_power: int = 4,
) -> Tuple[float, float, int]:
    """Return (lr_lower, lr_upper, num_forward_passes). η_c is in [lr_lower, lr_upper].

    Failure modes return (nan, nan, iters) so callers can propagate NaN: a NaN
    bracket makes the bisection condition `(1 - lr_lower/lr_upper) > tol`
    compare False and the bisection is skipped naturally.
    """
    with torch.no_grad():
        loss_init = loss_closure(batch)
    if not math.isfinite(loss_init):
        # Already-diverged run: no meaningful bracket exists.
        return float("nan"), float("nan"), 0

    @torch.no_grad()
    def update_and_compute_loss(lr: float) -> float:
        updates = virtual_step(model, optimizer, lr)
        with atParams(model, updates):
            return loss_closure(batch)

    lr = lr_guess
    iteration = 1
    loss_next = update_and_compute_loss(lr)
    direction = 1 if loss_next < loss_init else -1
    lr_lower, lr_upper = lr, lr

    MAX_ITERS = 100
    while iteration < MAX_ITERS:
        lr *= 2 ** direction
        loss_next = update_and_compute_loss(lr)
        iteration += 1
        if not math.isfinite(loss_next):
            # lr has been expanded into a divergent regime; every comparison
            # against a nan/inf loss is False, so continuing would silently
            # burn the whole iteration budget. Abort with a NaN bracket.
            return float("nan"), float("nan"), iteration
        if direction == 1 and loss_next > loss_init:
            lr_lower, lr_upper = lr / 2, lr
            break
        if direction == -1 and loss_next < loss_init:
            lr_lower, lr_upper = lr, lr * 2
            break
    else:
        # Bracketing exhausted without observing a loss flip: η_c lies outside
        # the explored range (or the loss is flat along Δθ). Returning the
        # degenerate (lr_guess, lr_guess) would fabricate λ_c = 2/lr_guess.
        return float("nan"), float("nan"), iteration

    tol = 1 / (2 ** tol_power)
    while (1 - lr_lower / lr_upper) > tol and iteration < MAX_ITERS + 20:
        lr_mid = (lr_lower + lr_upper) / 2.0
        loss_mid = update_and_compute_loss(lr_mid)
        iteration += 1
        if loss_mid > loss_init:
            lr_upper = lr_mid
        else:
            lr_lower = lr_mid

    return lr_lower, lr_upper, iteration


def critical_sharpness(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    loss_closure: Callable[[Any], float],
    lr_guess: float = 1e-3,
    tol_power: int = 4,
) -> Tuple[float, float, int]:
    """Return (λ_c, η_c, num_forward_passes)."""
    lr_lower, lr_upper, n = measure_critical_lr(
        model, optimizer, batch, loss_closure, lr_guess=lr_guess, tol_power=tol_power
    )
    eta_c = 0.5 * (lr_lower + lr_upper)
    if math.isnan(eta_c):
        # Failed bracket: propagate NaN instead of inf (the `eta_c > 0` check
        # below is False for NaN and would fabricate lam_c = inf).
        lam_c = float("nan")
    elif eta_c > 0:
        lam_c = 2.0 / eta_c
    else:
        lam_c = float("inf")
    return lam_c, eta_c, n
