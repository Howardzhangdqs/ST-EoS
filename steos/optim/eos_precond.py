"""EoS-Precond — rank-1 curvature preconditioning along the FREE top eigenvector.

The one surviving *plant-action* optimizer design enabled by this project's own
findings (F4, F3). The "predictor-not-actuator" negative result (F12/F15) showed
that *controller* actions (capping ‖W⟩/‖γ⟩, scaling η from a proxy) cannot
escape the EoS equilibrium λ_max → 2/η, because the plant's DC gain is −1 and
cancels them. Only *plant* actions (modifying the gradient itself) escape —
SAM escapes (F12: λ_max_final = 21 < 40 = 2/η), AdamW escapes.

EoS-Precond is a plant action: rank-1 natural-gradient preconditioning along the
top Hessian eigenvector v_top. Two findings make it cheap enough to be practical:

  * F4 — the single-step update Δθ aligns with v_top to |cos| → 0.998 at EoS.
    So v_top is observable FOR FREE as the unit direction of last step's Δθ.
    No prior rank-1 method gets v_top without power iteration (~100 HVPs).
  * F3/F9 — the GD trajectory is ~1-D (PC1 = 84–99 %), so rank-1 is SUFFICIENT.
    No prior cheap method can get away with rank-1; here F3 says K = 1.

Cost: ONE Hessian-vector product per step (the Rayleigh quotient v̂ᵀH v̂ ≈ λ_max
along the free direction), ~2.5× vanilla GD — comparable to SAM.

Update (plain-GD base, mom = 0, η fixed):
    g_par  = (gᵀ v̂) v̂             (component along the sharp axis)
    g_perp = g − g_par
    λ̂      = v̂ᵀ H v̂  ≥ 0           (one HVP — the Rayleigh quotient)
    divisor = max(λ̂ / λ_targ, 1),   λ_targ = 2/η   (stability boundary)
    Δθ     = −η ( g_perp + g_par / divisor )

The max(·, 1) clamp makes the method SELF-GATING: when λ̂ ≤ 2/η (early training
or flat directions) divisor = 1 and EoS-Precond ≡ plain GD. It only fires once
the sharp direction actually crosses the stability boundary — exactly the EoS
condition. Pre-EoS, v̂ is misaligned with v_top, but then λ̂ underestimates λ_max
and the clamp keeps the method at SGD, so early misalignment is harmless.

Period-2 robustness: at EoS, Δθ alternates ±v_top. v̂ = unit(Δθ_{t−1}) captures
the *axis* (sign cancels in both the Rayleigh quotient of symmetric H and the
parallel projection g_par = (gᵀv̂)v̂), so the period-2 EoS oscillation does not
defeat the single-step lag. (Do NOT EMA-smooth raw Δθ — that cancels across the
period-2 cycle, the F4 methodology trap.)

Implemented as a SAM-style step FUNCTION (it needs model/loss/X/y for the HVP,
which a torch.optim.step() cannot access) wrapping a base SGD optimizer.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from ..sharpness.hessian import hvp, max_eigenvalue


def _param_list(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _dot(a, b) -> float:
    return float(sum((ai * bi).sum().item() for ai, bi in zip(a, b)))


def _norm(a) -> float:
    return math.sqrt(sum((ai * ai).sum().item() for ai in a))


class EoSPrecondState:
    """Carries the free top-eigenvector estimate v̂ (unit Δθ from the previous
    step) and exposes diagnostics (λ̂, alignment, divisor) for logging."""

    def __init__(self) -> None:
        self.v_hat: Optional[List[torch.Tensor]] = None  # unit-norm, one per param
        self.step_count: int = 0
        # diagnostics from the most recent step
        self.lam_hat: float = 0.0      # Rayleigh quotient v̂ᵀ H v̂ (cheap λ_max estimate)
        self.alpha: float = 0.0        # gᵀ v̂  (signed parallel gradient component)
        self.divisor: float = 1.0
        self.fired: bool = False       # True iff divisor > 1 (preconditioning active)
        self.cos_g_vhat: float = 0.0   # |gᵀ v̂| / ‖g‖  (alignment of grad with v_top)
        # re-acquisition bookkeeping (the "is migration a v̂-staleness artifact?" test)
        self._last_reacquire_step: int = -10**9
        self._n_reacquired: int = 0

    def metrics(self) -> dict:
        return dict(lam_hat=self.lam_hat, alpha=self.alpha, divisor=self.divisor,
                    fired=int(self.fired), cos_g_vhat=self.cos_g_vhat,
                    n_reacquired=self._n_reacquired)


def eos_precond_step(
    model: torch.nn.Module,
    base_optimizer: torch.optim.Optimizer,
    loss_fn,
    X,
    y,
    state: EoSPrecondState,
    lr: float,
    lambda_targ: Optional[float] = None,
    warmup: int = 0,
    reacquire_threshold: Optional[float] = None,
    reacquire_iters: int = 30,
    reacquire_cooldown: int = 50,
) -> float:
    """One EoS-Precond step. See module docstring.

    Args:
        model, base_optimizer, loss_fn, X, y: the usual training objects.
        state: an EoSPrecondState holding v̂ from the previous step.
        lr: base learning rate η (used only to set λ_targ = 2/η if not given).
        lambda_targ: sharpness ceiling along v_top (default 2/η).
        warmup: steps to act as plain SGD before activating (the self-gating
            clamp usually makes this redundant; kept for experimentation).
        reacquire_threshold: if set, re-estimate v_top by power iteration
            whenever the PREVIOUS step's |cos(g, v̂)| fell below this — the
            "migration vs staleness" diagnostic. None ⇒ never re-acquire
            (pure free-v̂ design; the falsification baseline).
        reacquire_iters: power-iteration HVPs per re-acquisition (warm-started
            from the current v̂, so 20–40 suffices).
        reacquire_cooldown: min steps between re-acquisitions (bounds cost).

    Returns: the loss value (float) at θ_t (before the step).
    """
    if lambda_targ is None:
        lambda_targ = 2.0 / lr

    params = _param_list(model)
    base_optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    loss_val = float(loss.item())

    active = (state.v_hat is not None) and (state.step_count >= warmup)

    if active:
        # --- optional re-acquisition: refresh v_top when alignment dropped ---
        # (decides migration-vs-staleness: if a *correct* v_top still can't
        #  escape 2/η, the failure is structural, not a tracking artifact.)
        if (reacquire_threshold is not None
                and state.step_count - state._last_reacquire_step >= reacquire_cooldown
                and state.cos_g_vhat < reacquire_threshold):
            _, v_fresh = max_eigenvalue(model, loss_fn, X, y,
                                        n_iter=reacquire_iters,
                                        return_vec=True, v_init=state.v_hat)
            state.v_hat = v_fresh
            state._last_reacquire_step = state.step_count
            state._n_reacquired += 1

        v = state.v_hat                          # unit-norm list, one per param
        grads = [p.grad for p in params]
        gnorm = _norm(grads)
        alpha = _dot(grads, v)                   # gᵀ v̂  (signed)
        # λ̂ = v̂ᵀ H v̂ via ONE Hessian-vector product (Pearlmutter double-backprop)
        Hv = hvp(model, loss_fn, X, y, v, params)
        lam_hat = _dot(v, Hv) / max(_dot(v, v), 1e-30)
        lam_hat = max(lam_hat, 0.0)              # PSD floor (numerical guard)
        divisor = max(lam_hat / lambda_targ, 1.0)
        fired = divisor > 1.0 + 1e-9
        # record diagnostics (computed on the ORIGINAL gradient, before in-place edit)
        state.lam_hat = lam_hat
        state.alpha = alpha
        state.divisor = divisor
        state.fired = fired
        state.cos_g_vhat = abs(alpha) / max(gnorm, 1e-30)
        # in-place gradient edit: g_eff = g − α v̂ (1 − 1/divisor)
        # ⇒ Δθ = −η ( g_perp + g_par / divisor ), since g − g_par·(1−1/d) = g_perp + g_par/d
        if fired:
            c = alpha * (1.0 - 1.0 / divisor)
            with torch.no_grad():
                for p, vi in zip(params, v):
                    p.grad.sub_(c * vi)
    else:
        state.divisor = 1.0
        state.fired = False

    # snapshot θ to compute the actual Δθ (faithful to F4: v̂ ← unit Δθ)
    with torch.no_grad():
        before = [p.detach().clone() for p in params]

    base_optimizer.step()  # plain SGD reads the (possibly edited) p.grad

    with torch.no_grad():
        deltas = [p - b for p, b in zip(params, before)]
    dnorm = _norm(deltas)
    if dnorm > 1e-14:
        state.v_hat = [d / dnorm for d in deltas]   # unit Δθ → next step's free v̂
    # else: keep previous v_hat (degenerate step)
    state.step_count += 1
    return loss_val
