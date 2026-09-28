"""AdamW with `_compute_update` (ported from Meta scalable-curvature reference,
single-process). Exposing `_compute_update` keeps it compatible with the
critical-sharpness `measure_critical_lr` line search used elsewhere."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        new_step = state["step"] + 1
        bc1 = 1 - beta1 ** new_step
        bc2 = 1 - beta2 ** new_step
        new_exp_avg = exp_avg * beta1 + grad * (1 - beta1)
        new_exp_avg_sq = exp_avg_sq * beta2 + (grad ** 2) * (1 - beta2)
        numer = new_exp_avg / bc1
        denom = (new_exp_avg_sq / bc2).sqrt() + group["eps"]
        lr = group["lr"] if lr_override is None else lr_override
        delta = -lr * numer / denom
        if group["weight_decay"] != 0:
            delta = delta - lr * group["weight_decay"] * param
        if not virtual:
            state["step"] = new_step
            exp_avg.copy_(new_exp_avg)
            exp_avg_sq.copy_(new_exp_avg_sq)
        return delta

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                delta = self._compute_update(p, p.grad, self.state[p], group)
                p.add_(delta)
        return loss
