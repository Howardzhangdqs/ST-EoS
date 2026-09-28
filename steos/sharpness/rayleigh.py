"""Single-HVP Rayleigh-quotient sharpness proxies for λ_max (Candidate 1).

Two probe directions, each costing one Hessian-vector product:
  λ̂_grad   = gᵀHg / gᵀg         (gradient-direction Rayleigh quotient; ≡ λ_c)
  λ̂_update = ΔθᵀHΔθ / ΔθᵀΔθ     (update-direction Rayleigh quotient — the candidate)

Both are lower bounds on λ_max (Rayleigh quotient ≤ λ_max), tight iff the probe
aligns with v_top. At EoS, |cos(Δθ, v_top)| → 1 (Finding 4) so λ̂_update → λ_max;
pre-EoS (alignment ~0.1) it underestimates. Under momentum Δθ ≠ −ηg, so λ̂_update
can be a tighter λ_max monitor than λ̂_grad (≡ λ_c, det-r ≈ +0.49).

Positioned vs Roulet et al. NeurIPS 2024 (CDAT) and Kalra et al. 2026, who use
uᵀHu / directional sharpness for STEP-SIZE TUNING; here we frame it as a 1-HVP
λ_max MONITOR and characterize the |cos(·, v_top)| validity regime.

Consistency contract (see oracle review pitfall #1/#2):
  * g, Δθ, Hg, HΔθ must ALL be on the SAME batch as the λ_max they predict —
    the deterministic eval/sharpness batch. We recompute g via autograd.grad on
    (X, y) and derive Δθ from a virtual_step that consumes THAT grad (training
    grad is saved/restored untouched). On full-batch MLP this is a no-op; on
    mini-batch CIFAR/ViT it is essential (else g=train-batch, Hg=eval-batch ⇒
    silent corruption).
  * Caller MUST put the model in eval() (deterministic BN/LN Hessian, matching
    the λ_max being validated) and, for attention models, wrap in
    sdpa_kernel([SDPBackend.MATH]) so double-backward works.

Cost: λ̂_update = 1 HVP (the deployed proxy). λ̂_grad = +1 HVP +1 fwd-bwd, a
validation-only comparator. λ_max itself (warm-started power iteration) is the
ground truth built once, offline.
"""
from __future__ import annotations

import time
from typing import Dict, List

import torch

from .critical_lr import virtual_step
from .hessian import _dot, _norm, hvp


def rayleigh_proxies(
    model,
    optimizer,
    loss_fn,
    X,
    y,
    v_top: List[torch.Tensor],
    params: List[torch.Tensor],
    lr_override=None,
    hvp_timer=None,
) -> Dict[str, float]:
    """Return single-HVP Rayleigh proxies for λ_max, all on batch (X, y).

    Args:
        model:      nn.Module at θ. MUST be in eval() (caller's responsibility).
        optimizer:  must expose `_compute_update` (steos.optim.SGD) + param_names.
        loss_fn:    callable(pred, y) -> scalar.
        X, y:       the SAME deterministic batch λ_max was measured on.
        v_top:      top eigenvector (list of tensors) from max_eigenvalue(return_vec=True).
        params:     the trainable param list (== list passed to max_eigenvalue).
        lr_override: lr for virtual_step. λ̂_update is lr-scale-invariant (Rayleigh
                    quotient), so the value only affects the absolute Δθ scale, not
                    the proxy; defaults to the optimizer's group lr.
        hvp_timer:  optional callable(which: "grad"|"update", seconds). When given,
                    receives the wall-clock duration of each estimator's HVP —
                    the per-measurement cost record for the one-HVP warm-start
                    baseline comparator. Default None ⇒ no timing, no behavior
                    change.

    Returns dict: lambda_hat_grad, lambda_hat_update, cos_grad_vtop, cos_update_vtop.
    """
    # --- save training grads; install eval-batch gradient as the consistent g ---
    saved = {p: (p.grad.detach().clone() if p.grad is not None else None) for p in params}
    try:
        # g on the eval batch (the surface λ_max lives on)
        out = model(X)
        loss = loss_fn(out, y)
        g = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
        g = [gi.detach() for gi in g]
        # virtual_step reads param.grad → install g there (momentum buffer untouched)
        with torch.no_grad():
            for p, gi in zip(params, g):
                p.grad = gi.clone()

        lr = lr_override if lr_override is not None else optimizer.param_groups[0]["lr"]
        updates = virtual_step(model, optimizer, lr)  # dict {name: Δθ}, no state mutation

        # align Δθ to `params` order (params == requires_grad named params, in order)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        dtheta: List[torch.Tensor] = [updates[n] for n in names]
        if len(dtheta) != len(params):
            raise RuntimeError(
                f"Δθ/param length mismatch: {len(dtheta)} vs {len(params)}; "
                "ensure params == [p for p in model.parameters() if p.requires_grad]"
            )

        # HVPs on the eval batch (hvp does its own forward + double-backward).
        # Optional per-HVP timing (see hvp_timer docstring) — no-op when None.
        _t0 = time.perf_counter() if hvp_timer is not None else 0.0
        Hg = hvp(model, loss_fn, X, y, g, params)
        if hvp_timer is not None:
            hvp_timer("grad", time.perf_counter() - _t0)
        _t0 = time.perf_counter() if hvp_timer is not None else 0.0
        Hd = hvp(model, loss_fn, X, y, dtheta, params)
        if hvp_timer is not None:
            hvp_timer("update", time.perf_counter() - _t0)

        gg = _dot(g, g)
        dd = _dot(dtheta, dtheta)
        vn = _norm(v_top)
        rec = {
            "lambda_hat_grad": _dot(g, Hg) / max(gg, 1e-30),
            "lambda_hat_update": _dot(dtheta, Hd) / max(dd, 1e-30),
        }
        if vn > 0:
            gn = max(_norm(g), 1e-30)
            dn = max(_norm(dtheta), 1e-30)
            rec["cos_grad_vtop"] = _dot(g, v_top) / (gn * vn)
            rec["cos_update_vtop"] = _dot(dtheta, v_top) / (dn * vn)
        else:
            rec["cos_grad_vtop"] = float("nan")
            rec["cos_update_vtop"] = float("nan")
        return rec
    finally:
        # restore the training gradient so the optimizer step is unaffected
        with torch.no_grad():
            for p in params:
                sg = saved[p]
                p.grad = sg.clone() if sg is not None else None
