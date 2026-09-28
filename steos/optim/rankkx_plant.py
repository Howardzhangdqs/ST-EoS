"""Rank-k plant with ARPACK sensing — the accurate rank-k interpolation
(review round 2, headline experiment).

Generalizes ``true_lmax_plant`` (rank-1, ARPACK-sensed) to rank k: the
controller senses the top-k eigenpairs of the SAME deterministic eval batch by
warm-started ARPACK every ``refresh_n`` steps and edits the gradient along all
k directions every step. At k=1 with ``edit="damp"`` the semantics are
IDENTICAL to the true_lmax plant (same divisor rule, same clip, same sensing),
so the existing truelmax runs anchor the k=1 point of the interpolation.

Per-step recipe:

  Sense (every refresh_n steps):
    (λ̂_1..λ̂_k, v̂_1..v̂_k) ← ARPACK top-k (tol, warm-start v0 = previous v̂_1)
    on the deterministic eval batch (or, with ``sense_current=True``, on the
    CURRENT training minibatch — the fixed-batch-confound control: is the
    rank-1 inertness an artifact of damping a direction that is not the
    current minibatch's sharp direction?). Model in eval mode with BN
    save/restore during HVPs.

  Correct (every step), damping mode (edit="damp"):
    d_i  = λ̂_i / λ_targ                       # λ_targ = 2/η (no max(·,1) floor)
    f_i  = clip(1 − 1/d_i, −1, 1)              # same clip as true_lmax
    g_eff = g − Σ_i f_i (g·v̂_i) v̂_i
  amplification mode (edit="amplify", strength α):
    g_eff = g + α Σ_i (g·v̂_i) v̂_i             # γ = −α < 0, unconditional
                                                   (same as the rank-1 amplify
                                                   arm, generalized over k)

  momentum SGD on g_eff.

Diagnostics (jsonl, one record per step): sensed spectrum, divisors, damping
factors, fire counts, alignments |g·v̂_i|/‖g‖, Ritz residual at refresh.

The ``metrics_outcome`` hook lets the training driver compute the spectral-
redistribution overlaps (controller directions vs. the outcome-measured
eigenvectors) without re-deriving state.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..sharpness.hessian import (
    _hvp_flat_chunked,
    _param_list,
    max_eigenvalue_arpack,
)
from .true_lmax_plant import restore_bn, snapshot_bn

__all__ = ["RankKXState", "rankkx_plant_step"]


def _slices(params):
    starts, ends, s = [], [], 0
    for p in params:
        starts.append(s)
        s += p.numel()
        ends.append(s)
    return starts, list(zip(starts, ends))


class RankKXState:
    """Carries the k warm-started eigenvector estimates + per-step diagnostics."""

    def __init__(
        self,
        k: int = 1,
        refresh_n: int = 20,
        lam_targ: float = 20.0,
        chunk_size: int = 1024,
        sense_tol: float = 1e-2,
        sense_k: Optional[int] = None,
        edit: str = "damp",            # "damp" | "amplify"
        amplify: float = 1.0,          # α when edit == "amplify"
        sense_current: bool = False,   # sense on the CURRENT minibatch (confound control)
    ):
        assert edit in ("damp", "amplify")
        self.k = int(k)
        self.refresh_n = int(refresh_n)
        self.lam_targ = float(lam_targ)
        self.chunk_size = int(chunk_size)
        self.sense_tol = float(sense_tol)
        # ARPACK sensing asks for a few extra vectors so the k edit directions
        # are well-separated from the rest of the spectrum (default k+1).
        self.sense_k = int(sense_k) if sense_k is not None else self.k + 1
        if self.sense_k < self.k:
            raise ValueError("sense_k must be >= k")
        self.edit = edit
        self.amplify = float(amplify)
        self.sense_current = bool(sense_current)

        # Working state
        self.V_flat: Optional[List[torch.Tensor]] = None   # k unit-norm flat vectors
        self.v1_flat: Optional[np.ndarray] = None          # ARPACK warm start
        self.lams_sensed: List[float] = [float("nan")] * self.k
        self.step_count = 0
        self.last_refresh_step = int(-1)
        self.last_ritz = float("nan")

        # Per-step diagnostics (latest step)
        self.divisors: List[float] = [float("nan")] * self.k
        self.dampings: List[float] = [float("nan")] * self.k
        self.cos_g_v: List[float] = [float("nan")] * self.k
        self.is_refresh_step = False
        self.n_fired = 0
        self.gnorm = float("nan")

    def metrics(self) -> dict:
        return dict(
            step=self.step_count,
            edit=self.edit,
            k=self.k,
            lams_sensed=list(self.lams_sensed),
            divisors=list(self.divisors),
            dampings=list(self.dampings),
            cos_g_v=list(self.cos_g_v),
            is_refresh_step=int(self.is_refresh_step),
            n_fired=self.n_fired,
            ritz=self.last_ritz if self.is_refresh_step else float("nan"),
            gnorm=self.gnorm,
            last_refresh_step=self.last_refresh_step,
            refresh_n=self.refresh_n,
            sense_current=int(self.sense_current),
        )


def _sense_topk_arpack(model, loss_fn, X, y, params, v0_flat, k, tol, chunk_size,
                       device, dtype):
    """ARPACK top-k in EVAL mode with BN save/restore. Returns
    (evals_descending, evecs_columns_descending, ritz_residual_of_top_1)."""
    model.eval()
    saved_bn = snapshot_bn(model)
    try:
        evals, evecs = max_eigenvalue_arpack(
            model, loss_fn, X, y, params, k=k, tol=tol, v0=v0_flat,
            chunk_size=chunk_size)
        lam1 = float(evals[0])
        v_flat = np.ascontiguousarray(evecs[:, 0])
        v_t = torch.from_numpy(v_flat).to(device).type(dtype)
        Hv_flat = _hvp_flat_chunked(model, loss_fn, X, y, v_t, params, chunk_size)
        denom = max(abs(lam1) * float(v_t.norm().item()), 1e-30)
        ritz = float((Hv_flat - lam1 * v_t).norm().item() / denom)
        return evals, evecs, ritz
    finally:
        restore_bn(saved_bn)


def rankkx_plant_step(
    model,
    base_optimizer,
    loss_fn,
    Xb,
    yb,
    X_eval,
    y_eval,
    state: RankKXState,
):
    """One rank-k ARPACK-sensed plant step. See module docstring.

    ``X_eval/y_eval`` are used only when ``state.sense_current`` is False (the
    deterministic eval batch); with ``sense_current=True`` the sensing Hessian
    is the CURRENT training minibatch (Xb, yb), in eval mode with BN frozen.
    """
    params = _param_list(model)
    device, dtype = params[0].device, params[0].dtype

    # (1) train-batch forward/backward → g
    base_optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(Xb), yb)
    loss.backward()
    loss_val = float(loss.item())
    grads_orig = [p.grad.detach().clone() for p in params]
    g_flat = torch.cat([g.reshape(-1) for g in grads_orig])
    gnorm0 = float(g_flat.norm().item())

    # (2) refresh the top-k subspace every refresh_n steps
    is_refresh = (state.V_flat is None) or (state.step_count % state.refresh_n == 0)
    if is_refresh:
        Xs, ys = (Xb, yb) if state.sense_current else (X_eval, y_eval)
        evals, evecs, ritz = _sense_topk_arpack(
            model, loss_fn, Xs, ys, params, v0_flat=state.v1_flat,
            k=state.sense_k, tol=state.sense_tol,
            chunk_size=state.chunk_size, device=device, dtype=dtype)
        model.train()  # _sense ran in eval; restore for the next train batch
        kk = state.k
        state.lams_sensed = [float(x) for x in evals[:kk]]
        evec_t = torch.from_numpy(np.ascontiguousarray(evecs[:, :kk])).to(device).type(dtype)
        state.V_flat = [evec_t[:, i].contiguous() for i in range(kk)]
        state.v1_flat = np.ascontiguousarray(evecs[:, 0])
        state.last_ritz = ritz
        state.last_refresh_step = state.step_count

    assert state.V_flat is not None, "V must be set after first refresh"

    # (3) per-direction edit coefficients
    if state.edit == "amplify":
        divisors = [float("nan")] * state.k
        dampings = [-state.amplify] * state.k
    else:
        divisors, dampings = [], []
        for lam in state.lams_sensed:
            d = lam / state.lam_targ
            f = 0.0 if abs(d) < 1e-30 else 1.0 - 1.0 / d
            f = max(-1.0, min(1.0, f))
            divisors.append(float(d))
            dampings.append(float(f))

    # (4) α_i = g·v̂_i (single batched host sync) + gradient edit
    dots = torch.stack([torch.dot(g_flat, vf) for vf in state.V_flat])
    alphas = dots.tolist()
    with torch.no_grad():
        for i, vf in enumerate(state.V_flat):
            c = dampings[i] * alphas[i]          # g_eff = g − Σ c_i v̂_i
            off = 0
            for p, g in zip(params, grads_orig):
                n = p.numel()
                g.sub_(c * vf[off:off + n].view(p.shape))
                off += n
        for p, g in zip(params, grads_orig):
            p.grad.copy_(g)

    # (5) diagnostics
    state.divisors = divisors
    state.dampings = dampings
    state.cos_g_v = [abs(a) / max(gnorm0, 1e-30) for a in alphas]
    state.is_refresh_step = bool(is_refresh)
    state.n_fired = sum(1 for f in dampings if f > 0.1)
    state.gnorm = gnorm0

    base_optimizer.step()
    state.step_count += 1
    return loss_val
