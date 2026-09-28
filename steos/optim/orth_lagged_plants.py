"""Two new plant-action controllers for the C4 / lagged-phase experiments.

Both run on the existing tanh MLP [20-256-256-1] teacher-student full-batch GD
setup (LR=0.05, 2/η=40, 3000 steps) — the same as ``rankk_sweep_experiment.py``.

== C4: orthogonal damping (``orth_damp``) ==
The DUAL of the C3 rank-1 plant. C3 damps the top-eigenvector component
g_∥ = (gᵀv̂)v̂ and keeps the orthogonal complement g_⊥. C4 does the opposite:
KEEP g_∥ (the sharp direction) and DAMP g_⊥ (everything else):

    Δθ = −η ( g_∥ + (1/d) g_⊥ ),   d = orth damping strength

d=1 ⇒ Δθ = −η (g_∥ + g_⊥) = −η g  (vanilla SGD — sanity).
d→∞ ⇒ only the rank-1 sharp direction survives; all orthogonal directions frozen.
v̂ = warm-started power iteration top eigenvector (3 sweeps/step, same sensing
as the C3 rank-1 plant — NOT a frozen v̂).

The experiment question: does strong orthogonal damping (large d) drive the
trajectory to diverge, or does the EoS self-stabilization cap remain robust?
Both outcomes are reportable — do NOT pre-preset.

== Lagged-phase plant (``lagged_plant``) ==
The C3 rank-1 plant (no-clamp, γ = 1−1/d, d = λ̂/(2/η)) with the SENSING vector
artificially lagged by k steps:

    k = 0:  v̂ = accurate warm-started power iteration (every step, tol-stopped)
            — the "accurate" variant. Expected: sharp-escape to λ_max ~263.
    k ≥ 1:  v̂ = unit(Δθ_{t−k})  (the actual update from k steps ago)
            — k=1 is the "free-lagged" variant (expected pin at ~40);
            k=2,3 are the new observation points.

A length-3 ring buffer of recent Δθ is maintained; k ≥ 1 reads the
appropriately lagged entry. The damping formula is IDENTICAL across k (only the
sensing vector source changes), so the k-sweep isolates sensing-lag as the
single control variable.

Both controllers are SAM-style step functions wrapping a base SGD optimizer:
edit ``p.grad`` in place, then ``base_optimizer.step()``. Deterministic
given torch+numpy seed (no RNG inside the step).
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import torch

from ..sharpness.hessian import _param_list, hvp, max_eigenvalue

__all__ = [
    "OrthDampState", "orth_damp_step",
    "LaggedPlantState", "lagged_plant_step",
]


# --------------------------------------------------------------------------- #
# flat-parameter helpers (one-shot sync per step — MLP is small)               #
# --------------------------------------------------------------------------- #
def _flatten(vec_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in vec_list])


def _unflatten(flat: torch.Tensor, params: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    s = 0
    for p in params:
        e = s + p.numel()
        out.append(flat[s:e].view(p.shape))
        s = e
    return out


def _build_graph_once(model, loss_fn, X, y, params):
    """Forward + first backward with create_graph=True. Returns (g_flat_with_graph,
    hv_flat_closure, loss_val). The graph is retained so multiple Pearlmutter HVPs
    reuse it (warm-started power iter). Mirrors eos_precond_rankk.py L192-205."""
    out = model(X)
    loss = loss_fn(out, y)
    loss_val = float(loss.item())
    g = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    g_flat = torch.cat([gi.reshape(-1) for gi in g])      # 1-D, carries graph

    def hv_flat(v_flat: torch.Tensor) -> torch.Tensor:
        gdv = torch.dot(g_flat, v_flat)
        Hv = torch.autograd.grad(gdv, params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in Hv])

    return g_flat, hv_flat, loss_val


# ============================================================================ #
# C4: orthogonal damping                                                        #
# ============================================================================ #
class OrthDampState:
    """Carries the warm-started top-eigenvector estimate v̂ (flat, unit-norm) for
    the C4 orthogonal-damping controller. Diagnostics from the latest step are
    exposed for logging."""

    def __init__(self, n_sweeps: int = 3) -> None:
        self.v_hat: Optional[torch.Tensor] = None   # flat unit-norm
        self.n_sweeps = int(n_sweeps)
        self.step_count = int(0)
        # diagnostics
        self.lam_hat = float("nan")    # Rayleigh quotient v̂ᵀH v̂
        self.cos_g_v = float("nan")    # |gᵀv̂| / ‖g‖

    def metrics(self) -> dict:
        return dict(lam_hat=self.lam_hat, cos_g_v=self.cos_g_v,
                    step_count=self.step_count)


def orth_damp_step(
    model,
    base_optimizer,
    loss_fn,
    X,
    y,
    state: OrthDampState,
    lr: float,
    d: float,
    n_sweeps: Optional[int] = None,
) -> float:
    """One C4 orthogonal-damping step. See module docstring.

    Δθ = −η ( g_∥ + (1/d) g_⊥ ),  g_∥ = (gᵀv̂)v̂,  g_⊥ = g − g_∥.
    v̂ = warm-started power-iter top eigenvector (n_sweeps sweeps/step, same
    sensing as the C3 rank-1 plant).

    Args:
        d: orthogonal damping strength. d=1 ⇒ vanilla SGD (sanity). d>1 damps
            the orthogonal complement; d<1 amplifies it.
        n_sweeps: power-iter sweeps/step (defaults to state.n_sweeps).

    Returns: loss value at θ_t (before the step).
    """
    if n_sweeps is None:
        n_sweeps = state.n_sweeps
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")

    params = _param_list(model)
    n_flat = sum(p.numel() for p in params)
    base_optimizer.zero_grad(set_to_none=True)

    # build loss-gradient graph once, reuse for all HVPs
    g_flat, hv_flat, loss_val = _build_graph_once(model, loss_fn, X, y, params)

    # --- warm-started power iteration (n_sweeps sweeps) ---
    if state.v_hat is None or state.v_hat.numel() != n_flat:
        v = torch.randn(n_flat, device=g_flat.device, dtype=g_flat.dtype)
        v = v / v.norm().clamp(min=1e-12)
    else:
        v = state.v_hat
    for _ in range(n_sweeps):
        Hv = hv_flat(v)
        lam = torch.dot(v, Hv)
        nrm = Hv.norm().clamp(min=1e-12)
        v = Hv / nrm
    state.v_hat = v.detach()
    # Rayleigh quotient with the converged v
    Hv = hv_flat(v)
    lam_hat = float(torch.dot(v, Hv).item())
    state.lam_hat = max(lam_hat, 0.0)

    # --- gradient edit: g_eff = g_∥ + (1/d) g_⊥ ---
    # g_∥ = (g·v̂) v̂ ;  g_⊥ = g − g_∥
    # ⇒ g_eff = g_∥ + (1/d)(g − g_∥) = (1/d) g + (1 − 1/d) g_∥
    with torch.no_grad():
        g_flat_detached = g_flat.detach()
        alpha = torch.dot(g_flat_detached, v)         # gᵀv̂ (scalar tensor)
        cos = float(abs(alpha.item()) / max(float(g_flat_detached.norm().item()), 1e-30))
        state.cos_g_v = cos
        # g_eff_flat = (1/d) g + (1 − 1/d) (g·v̂) v̂
        g_eff_flat = (1.0 / d) * g_flat_detached + (1.0 - 1.0 / d) * alpha * v
        # write into p.grad
        for p, gi in zip(params, _unflatten(g_eff_flat, params)):
            p.grad = gi.clone()

    base_optimizer.step()
    state.step_count += 1
    return loss_val


# ============================================================================ #
# Lagged-phase plant                                                            #
# ============================================================================ #
class LaggedPlantState:
    """Carries (a) the warm-started accurate v̂ for k=0 mode, (b) a length-k ring
    buffer of recent unit-Δθ for k≥1 ``dtheta`` mode, (c) a ring buffer of
    accurate top-eigenvector estimates û(·) for ``eigvec`` mode, and (d) per-step
    diagnostics.

    sensing_mode:
      ``"dtheta"`` (default): k≥1 uses unit(Δθ_{t−k}) as v̂ — the FREE lagged
        variant. Δθ is an INACCURATE proxy for u₁ (gradient/update direction,
        not eigenvector); the lag-vs-inaccuracy effects are CONFOUNDED.
      ``"eigvec"``: k≥1 uses û(t−k), the accurate power-iter top eigenvector
        from k steps ago. This ISOLATES the lag effect: the eigenvector is
        ACCURATE for its time, just stale. Every step still computes û(t) via
        power iter (like k=0 accurate) and pushes it to the buffer; the
        correction uses the lagged entry. Cost ≈ k=0 accurate (~50 HVPs/step).
    """

    def __init__(self, lag_k: int, accurate_iters: int = 100,
                 accurate_tol: float = 1e-4,
                 sensing_mode: str = "dtheta") -> None:
        assert lag_k >= 0
        assert sensing_mode in ("dtheta", "eigvec")
        self.lag_k = int(lag_k)
        self.accurate_iters = int(accurate_iters)
        self.accurate_tol = float(accurate_tol)
        self.sensing_mode = str(sensing_mode)
        # accurate-mode working state (k=0 AND eigvec mode use power iter)
        self.v_hat_flat: Optional[torch.Tensor] = None   # warm-start for power iter
        self.v_hat: Optional[List[torch.Tensor]] = None  # list form (for hvp)
        # lagged-mode ring buffer of unit-Δθ (dtheta mode); entries are FLAT tensors
        self.delta_ringbuffer: deque = deque(maxlen=max(1, lag_k) + 1)
        # eigvec-mode ring buffer of accurate û(·) (flat tensors, unit-norm)
        #   maxlen is generous: we only need k+1 entries but keep 4 to avoid
        #   surprise when k is re-parameterized.
        self.eig_ringbuffer: deque = deque(maxlen=max(4, lag_k + 2))
        # diagnostics
        self.step_count = int(0)
        self.lam_hat = float("nan")     # Rayleigh quotient of the chosen v̂
        self.divisor = float("nan")
        self.gamma = float("nan")       # damping factor (1 − 1/d), clipped [-1,1]
        self.fired = False

    def metrics(self) -> dict:
        return dict(lam_hat=self.lam_hat, divisor=self.divisor, gamma=self.gamma,
                    fired=int(self.fired), step_count=self.step_count,
                    lag_k=self.lag_k, sensing_mode=self.sensing_mode)


def lagged_plant_step(
    model,
    base_optimizer,
    loss_fn,
    X,
    y,
    state: LaggedPlantState,
    lr: float,
    lambda_targ: Optional[float] = None,
    clamp: bool = True,
) -> float:
    """One lagged-phase plant step. See module docstring.

    Sensing:
      k = 0:  v̂ ← accurate power iteration (warm-started, tol-stopped, every step)
      k ≥ 1:  v̂ ← unit(Δθ_{t−k}) from the ring buffer (free-lagged)

    Correction:
      λ̂ = v̂ᵀH v̂  (one HVP)
      d = λ̂ / λ_targ,  λ_targ = 2/η
      γ = clip(1 − 1/d, −1, 1)         (NO max(·,1) floor on d itself)
      g_eff = g − γ (gᵀv̂) v̂
      Δθ = −η g_eff                     (plain SGD base)

    Args:
        clamp: if True (DEFAULT), apply the self-gating ``max(d, 1)`` floor —
            a direction at/below λ_targ gets divisor 1 (≡ plain SGD along that
            axis). This matches the EXISTING accurate / free-lagged variants
            (``eos_precond_step``) and is REQUIRED for the free-lagged k≥1
            cases to pin at 2/η rather than diverge: without the clamp, when
            λ̂ fluctuates around λ_targ (Rayleigh on the lagged Δθ under-
            estimates λ_max), γ flips to −1 and anti-damps the sharp direction,
            causing loss explosion at the EoS boundary (verified: step 543
            divergence). If False, run clamp-free (cap-removal semantics);
            stable only for accurate sensing (k=0).

    Returns: loss value at θ_t (before the step).
    """
    if lambda_targ is None:
        lambda_targ = 2.0 / lr

    params = _param_list(model)
    base_optimizer.zero_grad(set_to_none=True)

    # θ snapshot (needed for Δθ at the end, regardless of mode)
    with torch.no_grad():
        theta_before = _flatten([p.detach().clone() for p in params])

    # forward + backward (no create_graph — sensing HVP does its own fwd+bwd)
    out = model(X)
    loss = loss_fn(out, y)
    loss_val = float(loss.item())
    loss.backward()
    g_list: List[torch.Tensor] = []
    for p in params:
        assert p.grad is not None, "p.grad must be set after backward()"
        g_list.append(p.grad.detach().clone())
    g_flat = _flatten(g_list)

    # --- choose the sensing vector v̂ ---
    k = state.lag_k
    if state.sensing_mode == "eigvec":
        # ---- eigvec mode: ALWAYS compute û(t) via power iter, store, use lagged ----
        # This isolates lag: the eigenvector is ACCURATE for its time, just stale.
        # Cost ≈ k=0 accurate (~50 HVPs/step for power-iter convergence).
        if state.v_hat is None:
            _lam_cur, v_list = max_eigenvalue(
                model, loss_fn, X, y, n_iter=state.accurate_iters,
                tol=state.accurate_tol, return_vec=True)
        else:
            _lam_cur, v_list = max_eigenvalue(
                model, loss_fn, X, y, n_iter=state.accurate_iters,
                tol=state.accurate_tol, return_vec=True, v_init=state.v_hat)
        state.v_hat = v_list                       # warm-start for next step
        u_flat = _flatten(v_list).detach()
        state.eig_ringbuffer.append(u_flat)        # store û(t)

        # use k-step-lagged û(t−k) from the buffer. After appending, the buffer
        # holds [û(t−m), …, û(t)]. û(t−k) = buffer[-(k+1)]. During warm-up
        # (buffer has ≤ k entries), use the OLDEST available û.
        if len(state.eig_ringbuffer) > k:
            v_flat = state.eig_ringbuffer[-(k + 1)]
        else:
            v_flat = state.eig_ringbuffer[0]       # oldest available (warm-up)
        # λ̂ = v̂ᵀH(t) v̂ — Rayleigh quotient of the lagged eigenvector at the
        # CURRENT Hessian (1 HVP). This is the sharpness the plant "sees" along
        # its stale direction at the current θ.
        Hv = hvp(model, loss_fn, X, y, _unflatten(v_flat, params), params)
        Hv_flat = _flatten(Hv)
        state.lam_hat = float(torch.dot(v_flat, Hv_flat).item())
    elif k == 0:
        # accurate warm-started power iteration (tol-stopped). We do NOT reuse
        # the train-graph HVP — instead use max_eigenvalue's own hvp loop, which
        # builds a fresh graph each sweep (correct + simple; MLP is small).
        if state.v_hat is None:
            lam_hat, v_list = max_eigenvalue(
                model, loss_fn, X, y, n_iter=state.accurate_iters,
                tol=state.accurate_tol, return_vec=True)
        else:
            lam_hat, v_list = max_eigenvalue(
                model, loss_fn, X, y, n_iter=state.accurate_iters,
                tol=state.accurate_tol, return_vec=True, v_init=state.v_hat)
        state.v_hat = v_list
        v_flat = _flatten(v_list)
        state.lam_hat = float(lam_hat)
    else:
        # lagged: pull unit(Δθ_{t−k}) from the ring buffer. If buffer doesn't
        # have k entries yet (early steps), fall back to Δθ_{t−1} (most recent
        # available) — this matches the free-lagged variant's first-step
        # behavior (v̂ undefined ⇒ use the just-computed Δθ).
        idx = min(k, len(state.delta_ringbuffer))
        if len(state.delta_ringbuffer) == 0:
            # very first step: no Δθ history ⇒ sense via one quick power iter
            lam_hat, v_list = max_eigenvalue(
                model, loss_fn, X, y, n_iter=20, tol=1e-3, return_vec=True)
            state.v_hat = v_list
            v_flat = _flatten(v_list)
            state.lam_hat = float(lam_hat)
        else:
            # idx-th most recent entry (1 = most recent, k = oldest available)
            v_flat = state.delta_ringbuffer[-idx]
            # Rayleigh quotient v̂ᵀH v̂ via one HVP
            Hv = hvp(model, loss_fn, X, y, _unflatten(v_flat, params), params)
            Hv_flat = _flatten(Hv)
            state.lam_hat = float(torch.dot(v_flat, Hv_flat).item())

    # --- rank-1 correction (IDENTICAL across k) ---
    assert v_flat is not None, "v_flat must be set by the sensing branch above"
    v_flat_detached = v_flat.detach()
    lam_hat = max(state.lam_hat, 0.0)
    raw_divisor = lam_hat / lambda_targ
    if clamp:
        # self-gating: a direction at/below λ_targ gets divisor 1 (≡ plain SGD
        # along that axis), preventing anti-damping when λ̂ under-estimates.
        divisor = max(raw_divisor, 1.0)
    else:
        divisor = raw_divisor
    if abs(divisor) < 1e-30:
        gamma = 0.0
    else:
        gamma = 1.0 - 1.0 / divisor
    gamma = max(-1.0, min(1.0, gamma))     # clip to [-1, 1]

    alpha = float(torch.dot(g_flat, v_flat_detached).item())   # gᵀv̂
    # g_eff = g − γ (gᵀv̂) v̂
    with torch.no_grad():
        g_eff_flat = g_flat - gamma * alpha * v_flat_detached
        for p, gi in zip(params, _unflatten(g_eff_flat, params)):
            p.grad = gi.clone()

    state.divisor = float(divisor)
    state.gamma = float(gamma)
    state.fired = bool(gamma > 0.1)

    # --- take the step, then push unit(Δθ) to the ring buffer ---
    base_optimizer.step()
    with torch.no_grad():
        theta_after = _flatten([p.detach().clone() for p in params])
        delta = theta_after - theta_before
        dn = float(delta.norm().item())
        if dn > 1e-14:
            state.delta_ringbuffer.append(delta / dn)
        # else: degenerate step, don't push (keeps the buffer aligned in time)

    state.step_count += 1
    return loss_val
