"""X-SAM (Duan et al. 2026, arXiv:2601.10251) — FAITHFUL variant (no clamp).

This is the *faithful* re-implementation of X-SAM that resolves a contradiction
between our paper and Duan et al. Our existing `xsam.py` adds a SELF-GATING
CLAMP `divisor = max(λ̂/(2/η), 1)` so the rank-1 correction only fires when
λ̂ > 2/η — and on real runs that clamp makes the correction inert (0/3000 fires
on CIFAR-100/ResNet-18). That inertness is an ARTIFACT of our clamp, NOT a
property of Duan et al.'s X-SAM. The real X-SAM uses a FIXED α, applied
UNCONDITIONALLY every step, sign-gated. This module implements exactly that.

Faithful to Duan et al. Algorithm 2 / Eq. 11, per step:

  1. g      = ∇L(θ)                              (first forward-backward)
  2. ε      = ρ · g / ‖g‖                          (SAM perturbation, global)
     g_adv  = ∇L(θ + ε)                            (second forward-backward)
  3. v, λ̂  = top eigenpair of H(θ+ε)              (warm-started power iter)
  4. g_xsam = g_adv − α · |⟨g_adv, v⟩| · v          (Eq. 11, NO clamp, sign-gated)
  5. θ ← θ − η · g_xsam                            (plain SGD reads g_xsam)

THE CORRECTION (Eq. 11), derived explicitly. With v unit-norm and the signed
parallel projection coefficient α_g = ⟨g_adv, v⟩:

    g_xsam = g_adv − α · sign(α_g) · (α_g · v)
           = g_adv − α · |α_g| · v                         (since sign(x)·x = |x|)

We use the right-hand form: it is branch-free and numerically identical. There
is NO max(·,1), NO λ̂ threshold, NO divisor. α is a FIXED scalar (default 0.2).
The correction fires EVERY step, unconditionally.

Semantics (decompose g_adv = g_∥ + g_⊥ along/orthogonal to v):
  * α_g > 0 ⇒ g_xsam = g_⊥ + (1 − α) g_∥   (attenuate the toward-v component)
  * α_g < 0 ⇒ g_xsam = g_⊥ + (1 + α) g_∥   (amplify  the away-from-v component)
In both cases the SGD update −η·g_xsam moves AWAY from +v: a positive along-v
descent component is shrunk, a negative (away-from-v) one is amplified. This is
exactly the "push off the top eigenvector" plant action, sign-gated so it never
inverts the gradient.

NORMALIZATION CHOICE (documented + defensible): Duan et al. Algorithm 2
normalizes ĝ = g_adv / ‖g_adv‖ before the correction (their Eq. 9). However
their convergence theorem (5.1) analyzes the UN-normalized projector
P_t = I − α·v vᵀ acting on the RAW gradient, and the normalized form makes the
effective step size (and hence η) non-comparable to plain SAM/SGD. We therefore
implement the UN-NORMALIZED form (the theorem's P_t), so η = 0.1 means the same
thing for SGD, SAM, and X-SAM-faithful. This is the apples-to-apples comparison
our paper needs. We do NOT implement the normalized variant here.

Cost: 2 forward-backward (SAM) + ~5 warm-started HVPs (power iter) per step in
steady state (pi_iters_first on the first step to seed the warm start). The
warm-start v is carried in XSAMFaithfulState across steps (v_top migrates
slowly), exactly mirroring the mechanics of steos/optim/xsam.py.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..sharpness.hessian import max_eigenvalue
from .eos_precond import _dot, _norm, _param_list  # reuse, do not reimplement


class XSAMFaithfulState:
    """Carries the warm-start top-eigenvector estimate v across steps and exposes
    the faithful-variant diagnostics (λ̂, signed α_g, alignment, correction
    magnitude). `fired` is ALWAYS True (the correction is unconditional); the
    field is kept for schema parity with the clamped XSAMState so analyzers can
    treat both variants uniformly."""

    def __init__(self) -> None:
        self.v: Optional[List[torch.Tensor]] = None  # unit-norm list, one per param
        self.step_count: int = 0
        self.n_fired: int = 0  # cumulative steps where the correction applied (= step_count)
        # diagnostics from the most recent step
        self.lam_hat: float = 0.0       # Rayleigh quotient vᵀ H v (cheap λ_max estimate)
        self.alpha_g: float = 0.0       # ⟨g_adv, v⟩   (SIGNED parallel component)
        self.cos_g_v: float = 0.0       # |⟨g_adv, v⟩| / ‖g_adv‖   (alignment of g_adv with v)
        self.corr_norm: float = 0.0     # α · |α_g| · ‖v‖ = α·|α_g|   (correction magnitude, ‖v‖=1)
        self.fired: bool = True         # ALWAYS True (unconditional) — schema parity
        self.alpha_coeff: float = 0.0   # the fixed α used this step (for the analyzer)

    def metrics(self) -> dict:
        return dict(
            lam_hat=self.lam_hat,
            alpha_g=self.alpha_g,
            cos_g_v=self.cos_g_v,
            corr_norm=self.corr_norm,
            fired=int(self.fired),
            n_fired=self.n_fired,
            alpha_coeff=self.alpha_coeff,
        )


def xsam_faithful_step(
    model: torch.nn.Module,
    base_optimizer: torch.optim.Optimizer,
    loss_fn,
    X,
    y,
    state: XSAMFaithfulState,
    lr: float,
    rho: float = 0.05,
    alpha: float = 0.2,
    pi_iters: int = 5,
    pi_iters_first: int = 25,
) -> float:
    """One FAITHFUL X-SAM step (Duan et al. Eq. 11, NO clamp). See module docstring.

    Args:
        model, base_optimizer, loss_fn, X, y: the usual training objects.
        state: an XSAMFaithfulState holding the warm-start v from the previous step.
        lr: base learning rate η (plain SGD step size; kept for API parity / λ_targ
            derivations but NOT used to gate the correction — there is no gate).
        rho: SAM perturbation radius (matches the SAM baseline / xsam.py default 0.05).
        alpha: the FIXED correction strength α (Duan et al. default 0.2). Applied
            unconditionally every step, sign-gated.
        pi_iters: warm-started power-iteration HVPs per step in steady state.
        pi_iters_first: one-time burn-in on the first step (state.v is None).

    Returns: the loss value (float) at θ_t (before the step).
    """
    params = _param_list(model)

    # (i) gradient g at θ
    base_optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    loss_val = float(loss.item())
    grads_orig = [p.grad.detach() for p in params]

    # guard: degenerate (near-zero) gradient — fall back to a plain SGD step on g.
    gnorm0 = _norm(grads_orig)
    if gnorm0 < 1e-12:
        with torch.no_grad():
            for p, g in zip(params, grads_orig):
                p.grad.copy_(g)
        base_optimizer.step()
        state.step_count += 1
        state.n_fired += 0  # no correction applied this (degenerate) step
        state.fired = True
        state.alpha_coeff = alpha
        return loss_val

    # (ii) SAM perturbation: ε = ρ·g/‖g‖ (gradient-ASCENT, ALL eigendirs); θ ← θ+ε.
    orig = [p.detach().clone() for p in params]
    scale = rho / (gnorm0 + 1e-12)
    with torch.no_grad():
        for p in params:
            p.add_(p.grad * scale)  # ASCENT perturbation (matches xsam.py L121-131)

    # (iii) recompute gradient g_adv at θ + ε
    base_optimizer.zero_grad(set_to_none=True)
    loss_pert = loss_fn(model(X), y)
    loss_pert.backward()
    # model is now at θ+ε; p.grad = g_adv
    grads = [p.grad.detach().clone() for p in params]

    # (iv) warm-started power iteration for v and λ̂ at the CURRENT model point
    # (θ+ε), consistent with where g_adv is taken. First step (state.v is None):
    # longer burn-in to seed the warm start; thereafter ~pi_iters HVPs/step.
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

    # (v) THE FAITHFUL CORRECTION (Eq. 11): g_xsam = g_adv − α·|⟨g_adv,v⟩|·v
    #     NO clamp, NO divisor, NO λ̂ threshold. Fixed α, sign-gated, unconditional.
    gnorm = _norm(grads)
    alpha_g = _dot(grads, v)                 # ⟨g_adv, v⟩   (SIGNED; ‖v‖=1)
    # ---- the correction line (paste-able) ----
    c = alpha * abs(alpha_g)                  # α·|α_g|·‖v‖ with ‖v‖=1
    g_xsam = [ga - c * vi for ga, vi in zip(grads, v)]
    # -------------------------------------------

    # diagnostics
    state.alpha_g = alpha_g
    state.cos_g_v = abs(alpha_g) / max(gnorm, 1e-30)
    state.corr_norm = c
    state.fired = True
    state.alpha_coeff = alpha

    # (vi) restore θ (keep the corrected perturbed-point gradient g_xsam)
    with torch.no_grad():
        for p, o in zip(params, orig):
            p.copy_(o)

    # (vii) write g_xsam into p.grad, then (viii) descend from θ
    with torch.no_grad():
        for p, g in zip(params, g_xsam):
            p.grad.copy_(g)

    base_optimizer.step()  # plain SGD reads the (corrected) p.grad
    state.step_count += 1
    state.n_fired += 1  # unconditional: correction applied every (non-degenerate) step
    return loss_val
