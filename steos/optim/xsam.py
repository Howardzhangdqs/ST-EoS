"""X-SAM ablation — isolates which component of X-SAM escapes the EoS threshold.

X-SAM (Duan et al. 2026, arXiv:2601.10251) combines TWO plant actions:

  (A) a GLOBAL SAM perturbation  ε = ρ·g/‖g‖  (Foret et al. 2021), and
  (B) a RANK-1 top-eigenvector gradient correction along v_top,
       g_corr = g − (g·v_top) v_top (1 − 1/divisor),
       divisor = max(λ̂ / λ_targ, 1),  λ̂ = v_topᵀ H v_top,  λ_targ = 2/η.

The §5 falsifiable prediction: X-SAM escapes the EoS equilibrium λ_max → 2/η
ONLY because of component (A) — the global SAM dual-ascent over ALL eigendirs.
The rank-1 correction (B) contributes nothing to escape, for exactly the same
"spectral-migration" reason our own EoS-Precond fails: dampening ONE eigendir
just migrates sharpness to the next un-dampened eigenvector, so λ_max stays at
2/η. (See steos/optim/eos_precond.py and Fig. 22 / analyze_multiseed_control.)

This module provides the two ablation variants under ONE step function:

  * xsam_full      (use_sam=True)  — (A)+(B), the full X-SAM.
  * xsam_rank1only (use_sam=False) — (B) only, ≈ EoS-Precond but with a
                                     power-iteration v_top (clean apples-to-
                                     apples with xsam_full, which MUST use
                                     power iteration for v_top).

Falsification logic (decided by analyze_xsam.py over multi-seed CIs):
  * if xsam_full ≈ sam (overlapping λ_max_final CIs) AND
        xsam_rank1only ≈ 2/η  ⇒  escape is ENTIRELY the global SAM component;
        the rank-1 correction adds nothing. §5 prediction CONFIRMED.
  * if xsam_full escapes MORE than sam (lower λ_max_final, disjoint CIs)
        ⇒  the rank-1 correction contributes; prediction WEAKENED.

Cost: xsam_full ≈ 2 forward-backward (SAM) + ~5 warm-started HVPs (power iter)
per step; xsam_rank1only ≈ 1 forward-backward + ~5 HVPs/step. Both power
iterations use a warm-start seed carried in XSAMState, so 5 HVPs/step suffices
in steady state (v_top migrates slowly); the very first step does a longer
burn-in to seed the warm start well.

Update (plain-SGD base, η fixed):
    λ_targ    = 2/η
    divisor   = max(λ̂ / λ_targ, 1)        (self-gating clamp)
    g_corr    = g − (g·v_top) v_top (1 − 1/divisor)
    Δθ        = −η g_corr

Implemented as a SAM-style step FUNCTION (it needs model/loss/X/y for the HVP),
wrapping a base SGD optimizer — same API shape as sam_step / eos_precond_step.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..sharpness.hessian import max_eigenvalue
from .eos_precond import _dot, _norm, _param_list  # reuse, do not reimplement


class XSAMState:
    """Carries the warm-start top-eigenvector estimate v_top across steps and
    exposes the same diagnostics as EoSPrecondState (λ̂, alignment, divisor)."""

    def __init__(self) -> None:
        self.v: Optional[List[torch.Tensor]] = None  # unit-norm list, one per param
        self.step_count: int = 0
        # diagnostics from the most recent step
        self.lam_hat: float = 0.0       # Rayleigh quotient vᵀ H v (cheap λ_max estimate)
        self.alpha: float = 0.0         # gᵀ v_top  (signed parallel gradient component)
        self.divisor: float = 1.0
        self.fired: bool = False        # True iff divisor > 1 (rank-1 correction active)
        self.cos_g_v: float = 0.0       # |gᵀ v_top| / ‖g‖  (alignment of grad with v_top)

    def metrics(self) -> dict:
        return dict(lam_hat=self.lam_hat, alpha=self.alpha, divisor=self.divisor,
                    fired=int(self.fired), cos_g_v=self.cos_g_v)


def xsam_step(
    model: torch.nn.Module,
    base_optimizer: torch.optim.Optimizer,
    loss_fn,
    X,
    y,
    state: XSAMState,
    lr: float,
    rho: float = 0.05,
    use_sam: bool = True,
    pi_iters: int = 5,
    pi_iters_first: int = 25,
    lambda_targ: Optional[float] = None,
) -> float:
    """One X-SAM ablation step. See module docstring.

    Args:
        model, base_optimizer, loss_fn, X, y: the usual training objects.
        state: an XSAMState holding the warm-start v_top from the previous step.
        lr: base learning rate η (used to set λ_targ = 2/η if not given).
        rho: SAM perturbation radius (only used when use_sam=True). Matches the
            paper's SAM baseline and sam_step's default (0.05).
        use_sam: toggle the GLOBAL SAM component.
            True  ⇒ xsam_full      (SAM perturbation + rank-1 correction).
            False ⇒ xsam_rank1only (rank-1 correction only).
        pi_iters: warm-started power-iteration HVPs per step in steady state.
        pi_iters_first: one-time burn-in on the first step (state.v is None) to
            seed the warm start well, so the rank-1 comparison is fair.
        lambda_targ: sharpness ceiling along v_top (default 2/η).

    Returns: the loss value (float) at θ_t (before the step).
    """
    if lambda_targ is None:
        lambda_targ = 2.0 / lr

    params = _param_list(model)

    # (i) gradient g at θ
    base_optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    loss_val = float(loss.item())
    grads_orig = [p.grad.detach() for p in params]

    if use_sam:
        # (ii) perturb θ → θ + ε,  ε = ρ·g/‖g‖  (SAM dual-ascent, ALL directions)
        gnorm = _norm(grads_orig)
        if gnorm < 1e-12:
            state.step_count += 1
            return loss_val
        orig = [p.detach().clone() for p in params]
        scale = rho / (gnorm + 1e-12)
        with torch.no_grad():
            for p in params:
                p.add_(p.grad * scale)  # ASCENT (gradient-ascent perturbation)

        # (iii) recompute gradient g_pert at θ + ε
        base_optimizer.zero_grad(set_to_none=True)
        loss_pert = loss_fn(model(X), y)
        loss_pert.backward()
        # model is now at θ+ε; p.grad = g_pert

    # the gradient we will correct & descend with:
    grads = [p.grad.detach().clone() for p in params]

    # (iv) warm-started power iteration for v_top and λ̂. HVPs run at the CURRENT
    # model point — θ+ε when use_sam (consistent with where g_pert is taken),
    # θ otherwise. First step (state.v is None): longer burn-in to seed the warm
    # start; thereafter ~pi_iters HVPs/step suffice since v_top migrates slowly.
    n_iter = pi_iters_first if state.v is None else pi_iters
    # Freeze BN running stats during the HVP power iteration: max_eigenvalue
    # does model(X) forwards in train mode that would otherwise update
    # running_mean/var (~n_iter extra updates/step), shifting the BN stats and
    # making the xsam trajectory diverge from a plain SAM trajectory (a confound
    # that invalidates the λ_max/acc comparison). Save/restore so the forwards
    # leave the running stats unchanged.
    _bn = [(m, m.running_mean.clone(), m.running_var.clone())
           for m in model.modules() if hasattr(m, "running_mean")]
    lam_hat, v = max_eigenvalue(model, loss_fn, X, y, n_iter=n_iter,
                                return_vec=True, v_init=state.v)
    for m, rm, rv in _bn:
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
    state.v = v
    lam_hat = max(lam_hat, 0.0)  # PSD floor (numerical guard)
    state.lam_hat = lam_hat

    # (v) rank-1 top-eigenvector correction: g_corr = g − (g·v) v (1 − 1/divisor)
    gnorm = _norm(grads)
    alpha = _dot(grads, v)                               # g·v_top  (signed)
    divisor = max(lam_hat / lambda_targ, 1.0)            # self-gating clamp
    fired = divisor > 1.0 + 1e-9
    state.alpha = alpha
    state.divisor = divisor
    state.fired = fired
    state.cos_g_v = abs(alpha) / max(gnorm, 1e-30)

    if fired:
        c = alpha * (1.0 - 1.0 / divisor)
        g_corr = [g - c * vi for g, vi in zip(grads, v)]
    else:
        g_corr = grads

    if use_sam:
        # (vi) restore θ (keep the corrected perturbed-point gradient)
        with torch.no_grad():
            for p, o in zip(params, orig):
                p.copy_(o)

    # (vii) write g_corr into p.grad, then (viii) descend from θ
    with torch.no_grad():
        for p, g in zip(params, g_corr):
            p.grad.copy_(g)

    base_optimizer.step()  # plain SGD reads the (corrected) p.grad
    state.step_count += 1
    return loss_val
