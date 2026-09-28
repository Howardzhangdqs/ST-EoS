"""WNSharpSGD — Mechanism C optimizer prototype (the novel contribution).

Hypothesis (from our findings): the last-hidden-layer weight norm ‖W_lh‖ is a
FREE per-step proxy for the sharpness λ_max (Finding 9: Pearson ≈ 0.82). We
operationalize that proxy as a cheap control signal — turning the
Li-et-al-2022 *analytical* observation into an *optimizer* (a contribution no
prior work makes, per our novelty sweep).

Modes
-----
  'none' : plain full-batch GD (baseline; identical to SGD).
  'proj' : C-Proj — after the SGD step, if ‖W_lh‖ exceeds a sharpness BUDGET
           (budget = budget_factor · ‖W_lh‖_init), project W_lh back onto the
           norm ball. Caps proxy sharpness directly.
  'lr'   : C-LR — scale the step LR down as ‖W_lh‖ rises above its EMA baseline
           (push back on progressive sharpening): η_t = η · (ema/‖W_lh‖)^β.

The class subclasses SGD so it inherits `_compute_update` (⇒ critical-sharpness
λ_c stays measurable on these runs for later cross-checks). Proxy params are
constrained/monitored by reference; for the MLP this is `fc2.weight`.
"""

from __future__ import annotations

from typing import Optional

import torch

from .sgd import SGD


class WNSharpSGD(SGD):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        mode: str = "proj",
        proxy_params: Optional[list[torch.Tensor]] = None,
        budget_factor: float = 2.0,
        lr_beta: float = 1.0,
        lr_ema: float = 0.99,
    ):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        assert mode in ("none", "proj", "lr"), f"unknown mode {mode}"
        self.mode = mode
        self.proxy_params = list(proxy_params) if proxy_params else []
        self.budget_factor = float(budget_factor)
        self.lr_beta = float(lr_beta)
        self.lr_ema = float(lr_ema)
        self._init_norms: Optional[list[float]] = None
        self._ema_norm: Optional[list[float]] = None
        self.last_lr_mult: float = 1.0  # exposed for logging

    @torch.no_grad()
    def _proxy_norms(self) -> list[float]:
        return [float(p.detach().pow(2).sum().sqrt().item()) for p in self.proxy_params]

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        norms_before = self._proxy_norms()
        if self._init_norms is None:
            self._init_norms = list(norms_before)
            self._ema_norm = list(norms_before)

        # --- C-LR: choose a per-step LR multiplier from the proxy ----------
        lr_mult = 1.0
        if self.mode == "lr" and self.proxy_params:
            mults = []
            for i, nb in enumerate(norms_before):
                base = max(self._ema_norm[i], 1e-8)
                mults.append((base / max(nb, 1e-8)) ** self.lr_beta)
            lr_mult = max(min(mults), 1e-3)  # most-constrained proxy wins
        self.last_lr_mult = lr_mult

        # --- the underlying SGD step (with the adapted LR) -----------------
        for group in self.param_groups:
            lr_use = group["lr"] * lr_mult
            for p in group["params"]:
                if p.grad is None:
                    continue
                delta = self._compute_update(p, p.grad, self.state[p], group, lr_override=lr_use)
                p.add_(delta)

        # --- C-Proj: cap proxy sharpness post-step -------------------------
        if self.mode == "proj":
            for i, p in enumerate(self.proxy_params):
                budget = self.budget_factor * self._init_norms[i]
                n = float(p.detach().pow(2).sum().sqrt().item())
                if n > budget and n > 0:
                    p.mul_(budget / n)

        # --- EMA update of the proxy baseline ------------------------------
        if self._ema_norm is not None:
            cur = self._proxy_norms()
            for i in range(len(self._ema_norm)):
                self._ema_norm[i] = self.lr_ema * self._ema_norm[i] + (1.0 - self.lr_ema) * cur[i]
        return loss
