"""True-λ_max rank-1 plant — the Reviewer-#1 actuator test (ST-EoS-PR-01).

This is the cap-removed, ARPACK-sensed rank-1 plant that the preregistration
(``runs/p0/PREREGISTRATION_true_lmax_plant.md``) specifies. The P0 diagnostics
established that the old C3 plant (``eos_precond_rankk`` with k=1) was inert
because its power-iter sensing catastrophically under-estimated λ_max
(P0 Task C: mean +45 % under-estimation, worst +65 %). That placed sensed
sharpness below 2/η, zeroing the damping. ARPACK sensing (P0 Task A) removes
the artifact: the true λ_max is ≈36 (median) on ResNet-18/CIFAR-100 at η=0.1,
comfortably above 2/η=20, so the actuator now operates with real force.

Per-step recipe (see preregistration §3):

  Sense (every N=20 steps, refresh):
    v̂_top, λ_max_sensed ← ARPACK top-1 on the deterministic eval batch B=512
                            (same fixed subset as P0 — seed=123). Warm-start v0
                            from the previous v̂_top (deterministic RNG on first
                            call). model.eval() + BN save/restore during the
                            HVPs (mirrors xsam_faithful.py L168-174).

  Correct (every step):
    divisor   = λ_max_sensed / λ_targ         # λ_targ = 2/η = 20
    damping   = clip(1 − 1/divisor, −1, 1)    # NO max(·,1) floor
    g_eff     = g − damping · (g·v̂_top) · v̂_top
    momentum SGD on g_eff                     # plain SGD.step()

  Outcome (independent, every 200 steps, logged by the training driver):
    ARPACK top-3 λ_max (tol=1e-6, k=3) on B=512 — used for the
    preregistration's escape / pin(a) / pin(b) classification.

The ``State`` carries the warm-start across steps and exposes per-step
diagnostics (divisor, damping, fire flag, Ritz residual) for the jsonl log.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..sharpness.hessian import (
    _hvp_flat_chunked,
    _param_list,
    _unflatten,
    max_eigenvalue_arpack,
)


# ----------------------------------------------------------------------------
# BN save/restore — copied verbatim from xsam_faithful.py L168-174. ARPACK's
# many HVP forwards would otherwise pollute BN running stats and decouple the
# plant trajectory from a comparable plain-SGD trajectory. model.eval() already
# prevents the updates; this is belt-and-braces against caller-mode mistakes.
# ----------------------------------------------------------------------------
def snapshot_bn(model):
    return [(m, m.running_mean.clone(), m.running_var.clone())
            for m in model.modules() if hasattr(m, "running_mean")]


def restore_bn(saved):
    for m, rm, rv in saved:
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)


class TrueLmaxState:
    """Carries warm-start v_top, last-sensed λ_max, and per-step diagnostics."""

    def __init__(
        self,
        refresh_n: int = 20,
        lam_targ: float = 20.0,
        chunk_size: int = 1024,
        sense_tol: float = 1e-2,
        sense_k: int = 2,
        amplify: Optional[float] = None,
    ):
        self.refresh_n = int(refresh_n)
        self.lam_targ = float(lam_targ)
        self.chunk_size = int(chunk_size)
        self.sense_tol = float(sense_tol)
        self.sense_k = int(sense_k)
        # AMPLIFICATION mode (sign-flip control of the 2-D sign map): when set,
        # the edit is the FIXED curvature-amplifying g_eff = g + amplify·(g·v̂)v̂
        # (γ = −amplify < 0), applied unconditionally; the divisor/damping
        # logic is bypassed (damping is logged as −amplify for the jsonl).
        # Sensing (warm-started ARPACK refresh) is IDENTICAL to the damping
        # plant so the two arms are directly comparable.
        self.amplify = amplify

        # Working state
        self.v_top: Optional[List[torch.Tensor]] = None       # unit-norm, list form
        self.v_top_flat: Optional[np.ndarray] = None          # for ARPACK warm-start
        self.lam_sensed: float = float("nan")
        self.step_count: int = 0
        self.last_refresh_step: int = -1
        self.last_ritz: float = float("nan")

        # Per-step diagnostics (latest step); NaN until first refresh
        self.divisor: float = float("nan")
        self.damping: float = float("nan")
        self.is_refresh_step: bool = False
        self.fire: bool = False           # damping > 0.1 (preregistration defn)
        self.gdotv: float = float("nan")
        self.gnorm: float = float("nan")
        self.n_fired: int = 0             # cumulative steps with fire=True

    def metrics(self) -> dict:
        """Per-step diagnostics for the jsonl log (preregistration §3)."""
        return dict(
            step=self.step_count,
            lam_sensed=self.lam_sensed,
            lam_targ=self.lam_targ,
            divisor=self.divisor,
            damping=self.damping,
            is_refresh_step=int(self.is_refresh_step),
            fire=int(self.fire),
            ritz=self.last_ritz if self.is_refresh_step else float("nan"),
            gdotv=self.gdotv,
            gnorm=self.gnorm,
            last_refresh_step=self.last_refresh_step,
            refresh_n=self.refresh_n,
            n_fired=self.n_fired,
        )


def sense_lmax_arpack(
    model, loss_fn, X, y, params, v0_flat, k, tol, chunk_size, device, dtype
):
    """One ARPACK sensing call in EVAL mode with BN save/restore. Returns
    (lam1, v_top_flat_unitnorm, ritz_residual).

    Ritz residual ‖Hv − λv‖/|λ| computed via ONE extra HVP after ARPACK
    returns. This is the staleness guard (preregistration §4.3): on refresh
    steps it should be small (ARPACK converged); if it grows between refreshes
    the v_top is stale and the effective-force rate flag fires.
    """
    model.eval()
    saved_bn = snapshot_bn(model)
    try:
        evals, evecs = max_eigenvalue_arpack(
            model, loss_fn, X, y, params, k=k, tol=tol, v0=v0_flat,
            chunk_size=chunk_size)
        lam1 = float(evals[0])
        v_flat = np.ascontiguousarray(evecs[:, 0])           # unit-norm column
        # one extra HVP for the Ritz residual
        v_t = torch.from_numpy(v_flat).to(device).type(dtype)
        Hv_flat = _hvp_flat_chunked(model, loss_fn, X, y, v_t, params, chunk_size)
        denom = max(abs(lam1) * float(v_t.norm().item()), 1e-30)
        ritz = float((Hv_flat - lam1 * v_t).norm().item() / denom)
        return lam1, v_flat, ritz
    finally:
        restore_bn(saved_bn)
        # NB caller restores train mode


def true_lmax_plant_step(
    model,
    base_optimizer,
    loss_fn,
    Xb,
    yb,
    X_eval,
    y_eval,
    state: TrueLmaxState,
):
    """One true-λ_max rank-1 plant step. See module docstring.

    Args:
        model:        nn.Module in TRAIN mode (caller's responsibility); the
                      train-batch forward must use train-mode BN to update
                      running stats normally.
        base_optimizer: a steos.optim.SGD (or any torch.optim.Optimizer with
                      momentum); we write the corrected gradient into p.grad
                      and call .step().
        loss_fn:      callable(pred, y) -> scalar.
        Xb, yb:       train-batch tensors.
        X_eval, y_eval: the deterministic B=512 sensing subset (matches P0).
        state:        TrueLmaxState.

    Returns: train-batch loss (float) BEFORE the correction is applied.
    """
    params = _param_list(model)
    device, dtype = params[0].device, params[0].dtype

    # (1) train-batch forward + backward → g in p.grad
    base_optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(Xb), yb)
    loss.backward()
    loss_val = float(loss.item())
    grads_orig: List[torch.Tensor] = []
    for p in params:
        assert p.grad is not None, "param.grad must be set after backward()"
        grads_orig.append(p.grad.detach().clone())

    # g norm (one host sync; logged)
    gnorm0 = float(torch.sqrt(
        torch.stack([(g * g).sum() for g in grads_orig]).sum()).item())

    # (2) refresh v_top + λ_max_sensed if it's time (or first step)
    is_refresh = (state.v_top is None) or (state.step_count % state.refresh_n == 0)
    if is_refresh:
        lam, v_flat, ritz = sense_lmax_arpack(
            model, loss_fn, X_eval, y_eval, params,
            v0_flat=state.v_top_flat,
            k=state.sense_k, tol=state.sense_tol,
            chunk_size=state.chunk_size, device=device, dtype=dtype)
        model.train()  # sense_ switched to eval; restore for next train batch
        state.lam_sensed = lam
        state.v_top_flat = v_flat
        state.v_top = _unflatten(
            torch.from_numpy(v_flat).to(device).type(dtype), params)
        state.last_ritz = ritz
        state.last_refresh_step = state.step_count

    # v_top is guaranteed non-None past the first refresh
    v = state.v_top
    assert v is not None, "v_top must be set after first refresh"

    # (3) correction — NO max(·,1) floor on the divisor (preregistration §3)
    if state.amplify is not None:
        # AMPLIFICATION mode: fixed γ = −α edit, bypassing the divisor rule
        divisor = float("nan")
        damping = -float(state.amplify)
    else:
        lam = state.lam_sensed
        divisor = lam / state.lam_targ
        if abs(divisor) < 1e-30:
            damping = 0.0
        else:
            damping = 1.0 - 1.0 / divisor
        damping = max(-1.0, min(1.0, damping))     # clip to [−1, 1]

    # ⟨g, v̂_top⟩ (v̂_top is unit-norm in flat space)
    gdotv_t = torch.zeros((), device=device, dtype=dtype)
    for g, vi in zip(grads_orig, v):
        gdotv_t = gdotv_t + (g * vi).sum()

    # g_eff = g − damping · (g·v̂_top) · v̂_top  (per-param; all on-device)
    with torch.no_grad():
        scale = damping * gdotv_t                 # scalar tensor
        g_eff = [g - scale * vi for g, vi in zip(grads_orig, v)]

    # (4) diagnostics
    state.divisor = float(divisor)
    state.damping = float(damping)
    state.is_refresh_step = bool(is_refresh)
    state.fire = True if state.amplify is not None else bool(damping > 0.1)
    state.gdotv = float(gdotv_t.item())
    state.gnorm = gnorm0
    if state.fire:
        state.n_fired += 1

    # (5) write g_eff → p.grad, take the base momentum-SGD step
    with torch.no_grad():
        for p, g in zip(params, g_eff):
            assert p.grad is not None
            p.grad.copy_(g)
    base_optimizer.step()
    state.step_count += 1
    return loss_val


__all__ = [
    "TrueLmaxState",
    "true_lmax_plant_step",
    "sense_lmax_arpack",
    "snapshot_bn",
    "restore_bn",
]
