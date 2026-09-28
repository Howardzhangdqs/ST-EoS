"""Custom SGD exposing `_compute_update` — the Meta scalable-curvature contract.

Why a custom optimizer? To measure critical sharpness lambda_c we must be able
to ask: "given an arbitrary learning rate eta, what update would the optimizer
take *without* mutating its state?" Stock torch optimizers cannot answer that.
This class mirrors facebookresearch/scalable-curvature/utils/optimizers.py so
that `virtual_step` / `measure_critical_lr` work unchanged in Phase 2.

For the canonical Edge-of-Stability experiments we use plain full-batch GD,
i.e. this SGD with momentum=0 and batch_size == dataset size.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch.optim.optimizer import ParamsT


class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def _compute_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        lr_override: Optional[float] = None,
        virtual: bool = False,
    ) -> torch.Tensor:
        """Return the delta that would be ADDED to `param` (i.e. Δθ)."""
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        lr = group["lr"] if lr_override is None else lr_override
        mom = group["momentum"]
        wd = group["weight_decay"]

        if mom != 0:
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]
            buf_next = mom * buf + grad
            update = buf_next
            if group["nesterov"]:
                update = grad + mom * buf_next
            if not virtual:
                buf.copy_(buf_next)
        else:
            update = grad

        delta = -lr * update
        if wd != 0:
            delta = delta - lr * wd * param
        return delta

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                delta = self._compute_update(param, param.grad, state, group)
                param.add_(delta)
        return loss


def register_param_names(optimizer: torch.optim.Optimizer, model: torch.nn.Module):
    """Inject `param_names` into each param_group so Meta's `virtual_step` can
    map updated tensors back to module names. Call once after building the
    optimizer. Returns the optimizer."""
    names = [n for n, _ in model.named_parameters()]
    for group in optimizer.param_groups:
        group["param_names"] = list(names[: len(group["params"])])
    return optimizer
