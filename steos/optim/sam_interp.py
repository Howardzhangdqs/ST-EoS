"""τ-interpolated SAM: g ← (1−τ)·g_lin + τ·g_SAM (review round 9).

SAM's gradient at the perturbed point and its first-order (linSAM) expansion
differ only by the NONLINEAR content of the perturb-and-requery:

    g_SAM = ∇L(θ + ρ·g/‖g‖)          (steos/optim/sam.py — re-query)
    g_lin = g + (ρ/‖g‖)·H_B g        (steos/optim/linsam.py — Taylor term)

This arm blends them with a single scalar τ ∈ [0, 1]: τ=1 recovers SAM
exactly, τ=0 recovers the linearized update (with BN matched — see below).
Varying τ traces how much of SAM's λ_max reduction survives as the nonlinear
re-query content is dialled out, complementing the ρ-sweep (round 5) with an
interpolation sweep at fixed ρ.

Faithfulness / endpoint contracts (validated by scripts/validate_tau_match.py):

  * TWO train-mode forwards per step (at θ and at θ+ε), in sam_step's order —
    BN running stats evolve exactly as in the sam / linsam_bnmatched arms
    (double-update), so every τ shares ONE normalization protocol.
  * ε = ρ·g/‖g‖ with sam_step's OWN per-parameter python-float accumulation
    order (see linsam_bnmatched_step) — bit-identical perturbation.
  * g_lin is constructed with linsam_bnmatched_step's exact arithmetic
    (v + c·Hg, c = ρ/math.sqrt(gg) from the fused flat sum).
  * Final gradient = (1−τ)·g_lin + τ·g_SAM as ONE tensor expression: in
    IEEE-754 arithmetic the endpoints are EXACT (τ=1 → 0·g_lin + g_SAM =
    g_SAM; τ=0 → g_lin + 0 = g_lin), so a 60-step deterministic run at
    τ=1 is bit-identical to the sam arm and τ=0 to linsam_bnmatched.
  * Degenerate ‖g‖→0 steps mirror sam_step/linsam_step: no perturbation, no
    second forward, no base step.
"""

from __future__ import annotations

import math

import torch

from steos.sharpness.hessian import _param_list


def sam_interp_step(model, loss_fn, X, y, optimizer, rho: float, tau: float,
                    state=None):
    """One τ-interpolated SAM step (see module docstring).

    state: optional LinSAMState (from steos.optim.linsam) filled with the
    linSAM-side diagnostics (gnorm, c, ‖Hg‖, cos(g, Hg), lam_step) so the
    per-step jsonl logging matches the linsam arms' schema.
    Returns the pre-step minibatch loss at θ (sam_step's convention).
    """
    base_optimizer = optimizer
    base_optimizer.zero_grad(set_to_none=True)
    params = _param_list(model)
    loss = loss_fn(model(X), y)                    # stats update 1 (= sam's fwd@θ)
    g = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = torch.cat([gi.reshape(-1) for gi in g])
    gg = float(g_flat.detach().pow(2).sum().item())
    if gg < 1e-24:                                  # matches sam's gnorm < 1e-12
        if state is not None:
            state.degenerate += 1
        return float(loss.item())
    v = g_flat.detach()                             # raw g(θ)
    gdotv = (g_flat * v).sum()                      # carries graph; d/dθ = H g
    Hv = torch.autograd.grad(gdotv, params, retain_graph=False)
    hg_flat = torch.cat([h.reshape(-1) for h in Hv]).detach()

    gnorm = math.sqrt(gg)                           # linsam arithmetic …
    c = rho / gnorm
    gHg = float((v * hg_flat).sum().item())
    hgn = float(hg_flat.pow(2).sum().sqrt().item())
    if state is not None:
        state.gnorm = gnorm
        state.c = c
        state.hg_norm = hgn
        state.corr_norm = c * hgn
        state.cos_g_hg = gHg / max(gnorm * hgn, 1e-30)
        state.lam_step = gHg / gg
    g_lin_flat = v + c * hg_flat                    # … = linsam_bnmatched's new_flat

    with torch.no_grad():
        # ε = ρ·g/‖g‖ with sam_step's own accumulation order (bit-identical).
        s2 = 0.0
        for gi in g:
            s2 += float(gi.detach().pow(2).sum().item())
        if s2 ** 0.5 < 1e-12:                       # sam_step's degenerate gate
            if state is not None:
                state.degenerate += 1
            return float(loss.item())
        scale = rho / (s2 ** 0.5 + 1e-12)
        orig = {n: p.detach().clone() for n, p in model.named_parameters()}
        o = 0
        for gi in g:                                # perturb by RAW g (= sam's ε)
            params[o].add_(gi.detach() * scale)     # ASCENT
            o += 1
    loss_p = loss_fn(model(X), y)                   # stats update 2 (sam's fwd@θ+ε)
    g_sam = torch.autograd.grad(loss_p, params)
    g_sam_flat = torch.cat([gi.detach().reshape(-1) for gi in g_sam])

    with torch.no_grad():
        for n, p in model.named_parameters():       # restore θ
            p.copy_(orig[n])
        # single-expression blend — EXACT at τ ∈ {0, 1} (see module docstring)
        g_final_flat = (1.0 - tau) * g_lin_flat + tau * g_sam_flat
        s = 0
        for p in params:
            e = s + p.numel()
            p.grad = g_final_flat[s:e].view(p.shape).clone()
            s = e
    base_optimizer.step()
    return float(loss.item())
