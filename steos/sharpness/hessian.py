"""Hessian max-eigenvalue estimation via power iteration with HVP.

Implements the classic sharpness measure of the Edge-of-Stability literature
(Cohen et al. ICLR 2021, and the `locuslab/edge-of-stability` reference):
  lambda_max = largest eigenvalue of the loss Hessian  H = ∇²L(θ).

Hessian-vector products use Pearlmutter's trick  Hv = ∇(g·v)  with
g = ∇L, computed via a second `torch.autograd.grad` call (double-backprop).
Converges in O(few × 10–100) HVPs; each HVP ≈ 2× a backward over the batch.

Also returns the top eigenvector `v_top` (optional) — used in Phase 3 to
measure the alignment of the per-step update Δθ with the sharpest direction
(the Damian et al. 2023 self-stabilization prediction).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


def _param_list(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _dot(a, b) -> float:
    return float(sum((ai * bi).sum().item() for ai, bi in zip(a, b)))


def _norm(a) -> float:
    return math.sqrt(sum((ai * ai).sum().item() for ai in a))


def _normalize(a):
    n = _norm(a)
    return [ai / n for ai in a] if n > 0 else [ai.clone() for ai in a]


def hvp(model, loss_fn, X, y, v, params) -> List[torch.Tensor]:
    """Hessian-vector product H v via Pearlmutter's trick. Returns detached list."""
    out = model(X)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gdotv = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for g, vi in zip(grads, v):
        gdotv = gdotv + (g * vi).sum()
    Hv = torch.autograd.grad(gdotv, params, retain_graph=False, create_graph=False)
    return [h.detach() for h in Hv]


def max_eigenvalue(
    model,
    loss_fn,
    X,
    y,
    n_iter: int = 200,
    tol: float = 1e-4,
    return_vec: bool = False,
    v_init=None,
    return_iters: bool = False,
):
    """Power-iteration estimate of the loss Hessian's largest eigenvalue.

    Args:
        model:   nn.Module at the point θ where curvature is measured.
        loss_fn: callable(pred, y) -> scalar.
        X, y:    batch over which L is defined (use the full dataset for EoS).
        n_iter:  max power-iteration steps.
        tol:     relative convergence tolerance on the eigenvalue.
        return_vec: also return the top eigenvector (list of tensors, unit norm).
        v_init:  optional initial vector (warm-start from previous step helps).
        return_iters: also return the number of HVPs used (steady-state warm-started
            cost — the value that justifies "cheap" measurement).

    Returns lambda_max (float), or (lambda_max, v_top) if return_vec, with a
    trailing n_hvp int appended when return_iters=True.
    """
    params = _param_list(model)
    v = v_init if v_init is not None else [torch.randn_like(p) for p in params]
    v = _normalize(v)

    lam = 0.0
    iters_used = 0
    for _ in range(n_iter):
        Hv = hvp(model, loss_fn, X, y, v, params)
        iters_used += 1
        lam_new = _dot(v, Hv)
        hn = _norm(Hv)
        if hn < 1e-14:
            lam = 0.0
            break
        v = [h / hn for h in Hv]
        converged = abs(lam_new - lam) <= tol * max(1.0, abs(lam_new))
        lam = lam_new
        if converged:
            break

    # Rayleigh quotient with the final eigenvector estimate (one more HVP)
    Hv = hvp(model, loss_fn, X, y, v, params)
    iters_used += 1
    lam = _dot(v, Hv) / max(_dot(v, v), 1e-30)

    out: list = [lam]
    if return_vec:
        out.append(v)
    if return_iters:
        out.append(iters_used)
    if len(out) == 1:
        return out[0]
    return tuple(out)


def top_k_eigenvalues(
    model,
    loss_fn,
    X,
    y,
    k: int = 12,
    n_iter: int = 100,
    v_init=None,
    return_vecs: bool = False,
):
    """Top-k Hessian eigenvalues via block power iteration with modified
    Gram-Schmidt orthonormalization. Returns a length-k list of floats in
    DESCENDING order (the migration-spectrum diagnostic for the rank-k sweep).

    Each of the ``n_iter`` sweeps performs k Hessian-vector products (one per
    direction), orthogonalizes against the already-converged directions of that
    sweep via MGS, and renormalizes — the standard symmetric block-power /
    subspace-iteration recipe. Converges fast when the spectrum has gaps; for a
    near-degenerate cluster the smaller eigenvalues are less accurate but the
    qualitative shape (where the curvature mass sits) is preserved.

    Implementation: the loss-gradient graph is built ONCE (``create_graph=True``)
    and reused for all ``k·n_iter`` second-backwards (Pearlmutter Hv), and the
    MGS runs with GPU-resident scalars — a single host sync at the very end.
    This makes a (k=12, n_iter=64) measurement ~10× faster than the naïve
    per-call ``hvp`` loop, which matters because the spectrum is recorded at
    every measurement step of the sweep.

    Args:
        model, loss_fn, X, y: measured at the current θ.
        k: number of top eigenpairs to estimate.
        n_iter: block-power sweeps (each costs k HVPs).
        v_init: optional warm-start, a list of k orthonormal vector-lists.
        return_vecs: also return the k converged orthonormal eigenvectors.

    Returns:
        list[float] of length k (descending), or ``(eigenvalues, V)`` if
        ``return_vecs`` (V is a list of k unit-norm vector-lists).
    """
    params = _param_list(model)

    # build the loss-gradient graph ONCE; reuse for every Hv (second-backward).
    out = model(X)
    loss = loss_fn(out, y)
    g = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    g_flat = torch.cat([gi.reshape(-1) for gi in g])        # 1-D, carries graph

    slices: list = []
    s = 0
    for p in params:
        e = s + p.numel()
        slices.append((s, e))
        s = e
    n_flat = s

    def hv_flat(v_flat):
        gdv = torch.dot(g_flat, v_flat)                     # scalar w/ graph
        Hv = torch.autograd.grad(gdv, params, retain_graph=True)  # tuple
        return torch.cat([h.reshape(-1) for h in Hv])       # 1-D

    # --- initialize k orthonormal flat vectors (MGS on random draws, or warm) -
    if v_init is not None and len(v_init) == k:
        V = [v.reshape(-1).clone() for v in v_init]
    else:
        V = []
        for _ in range(k):
            v = torch.randn(n_flat, device=g_flat.device, dtype=g_flat.dtype)
            for b in V:
                v = v - torch.dot(v, b) * b
            v = v / v.norm().clamp(min=1e-12)
            V.append(v)

    # --- warm-started block power iteration (flat; GPU-resident, zero syncs) --
    zero = g_flat.new_zeros(())
    lams_t: List[torch.Tensor] = [zero.clone() for _ in range(k)]  # Rayleigh quotients
    for _ in range(n_iter):
        V_new: List[torch.Tensor] = []
        for i in range(k):
            Hv = hv_flat(V[i])                              # 1-D (H v_i)
            lams_t[i] = torch.dot(V[i], Hv)                 # 0-dim Rayleigh quotient
            for b in V_new:                                 # MGS vs v_1..v_{i-1}
                Hv = Hv - torch.dot(Hv, b) * b
            V_new.append(Hv / Hv.norm().clamp(min=1e-12))
        V = V_new

    # ONE batched sync for all k eigenvalues
    lams = [max(x, 0.0) for x in torch.stack(lams_t).tolist()]

    # block power iteration yields approximately-descending eigenvalues; enforce.
    order = sorted(range(k), key=lambda i: -lams[i])
    lams = [lams[i] for i in order]
    V = [V[i] for i in order]
    if return_vecs:
        # convert flat vectors back to per-parameter lists
        V_out = [[v[a:b].view(p.shape) for p, (a, b) in zip(params, slices)] for v in V]
        return lams, V_out
    return lams


# ============================================================================
# ARPACK top-k Hessian eigenpairs (ground truth for the power-iter validation).
# =================================================================-----------

def _flatten(vec_list: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of tensors (any shape) into a 1-D tensor."""
    return torch.cat([v.reshape(-1) for v in vec_list])


def _unflatten(flat: torch.Tensor, params: List[torch.Tensor]) -> List[torch.Tensor]:
    """Inverse of `_flatten`: split a 1-D tensor back to per-param shapes (views)."""
    out: List[torch.Tensor] = []
    s = 0
    for p in params:
        e = s + p.numel()
        out.append(flat[s:e].view(p.shape))
        s = e
    return out


def _param_geometry(params: List[torch.Tensor]) -> Tuple[List[Tuple[int, int]], int]:
    """Return (slices, n_flat) where slices[i] = (start, end) into the flat vector."""
    slices: List[Tuple[int, int]] = []
    s = 0
    for p in params:
        e = s + p.numel()
        slices.append((s, e))
        s = e
    return slices, s


def _hvp_flat_chunked(
    model,
    loss_fn,
    X,
    y,
    v_flat: torch.Tensor,
    params: List[torch.Tensor],
    chunk_size: Optional[int],
) -> torch.Tensor:
    """Flat Hessian-vector product H v where H = ∇²L, L = loss_fn(model(X), y)
    with reduction='mean' over the full batch of size B.

    When ``chunk_size`` is None or >= B, a single forward + double-backward is
    used (the standard `hvp` path). When smaller, the batch is split into chunks
    of (at most) ``chunk_size`` and the per-chunk Hv is accumulated with weight
    c/B — algebraically exact for mean-reduction losses (each chunk's mean is
    multiplied by its fraction of the full batch, recovering the full-batch mean
    Hessian). This is what lets ARPACK measure λ_max on B=51200 (full CIFAR-100
    train) without OOM: only one chunk's activations + double-backward graph
    live on the device at a time.
    """
    B = X.shape[0]
    if chunk_size is None or chunk_size >= B:
        v_list = _unflatten(v_flat, params)
        Hv = hvp(model, loss_fn, X, y, v_list, params)
        return _flatten(Hv)

    Hv_flat = torch.zeros_like(v_flat)
    v_list = _unflatten(v_flat, params)  # const across chunks → unflatten once
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        c = e - s
        Hv_chunk = hvp(model, loss_fn, X[s:e], y[s:e], v_list, params)
        # full-batch mean loss = Σ_chunks (c/B) · (1/c Σ_{i in chunk} l_i)
        # ⇒ H v = Σ_chunks (c/B) · Hv_chunk
        Hv_flat = Hv_flat + (float(c) / float(B)) * _flatten(Hv_chunk)
    return Hv_flat


def max_eigenvalue_arpack(
    model,
    loss_fn,
    X,
    y,
    params: List[torch.Tensor],
    k: int = 3,
    tol: float = 1e-6,
    v0: Optional[np.ndarray] = None,
    which: str = "LA",
    chunk_size: Optional[int] = None,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """ARPACK top-k eigenpair of the loss Hessian via scipy.sparse.linalg.eigsh
    wrapped around our Pearlmutter HVP. This is the GROUND-TRUTH λ_max used to
    validate the cheaper power-iteration estimator and to characterize the
    spectrum of vanilla SGD on ResNet-18/CIFAR-100 (the central P0 diagnostic).

    Reference: locuslab/edge-of-stability src/utilities.py lanczos() — same idea
    (Implicitly Restarted Arnoldi on a Hessian-vector matvec), ARPACK's eigsh
    is the production-grade implementation.

    Args:
        model, loss_fn, X, y: the loss L(model(X), y) whose Hessian is measured.
        params: list of trainable tensors (== _param_list(model)).
        k: number of top eigenpairs requested. ARPACK requires k < n.
        tol: relative eigenvalue tolerance passed to eigsh.
        v0: warm-start vector (1-D np.float32/float64 of length n). None ⇒
            deterministic RNG-seeded gaussian (deterministic for reproducibility
            across opt trajectories; the same v0 ⇒ same convergence path ⇒
            directly-comparable λ_max across optimizers).
        which: 'LA' (largest algebraic) by default — the CE Hessian can have
            NEGATIVE eigenvalues early in training, so 'LM' (largest magnitude)
            could spuriously return a large negative eigenvalue. Use 'LA'.
        chunk_size: see `_hvp_flat_chunked`. None ⇒ single forward (OOMs above
            B≈2–4k on a 16 GB GPU for ResNet-18 / CIFAR-100). 2048 is a safe
            default that keeps the double-backward graph small.
        maxiter, ncv: passed through to eigsh (None ⇒ ARPACK defaults).
            ncv is the main cost lever: the default ncv=20 forces one full
            Arnoldi restart cycle = k+ncv+1 ≈ 21+ matvecs per solve even when
            warm-started; measured on ResNet-18/CIFAR-10 (B=512): k=3/ncv=20
            = 21 matvecs, k=3/ncv=8 = 17, k=1/ncv=8 = 9 (~2.6× cheaper) with
            rel err ≤ 1e-7 vs the default config.

    Caller contract (CRITICAL — silent corruption otherwise):
      * model MUST be in eval() — BatchNorm in train mode would (a) update
        running_mean/var inside the ARPACK matvec (many forwards ⇒ many phantom
        updates) and (b) compute a different Hessian (batch stats vs running
        stats). Following the xsam_faithful.py pattern, also save/restore
        running_mean/var around the call as belt-and-braces.
      * torch.use_deterministic_algorithms(True) is honored by the matvec
        (forward + double-backward); ARPACK itself is CPU/numpy, fully
        deterministic given v0.

    Returns:
        (evals, evecs): both DESCENDING (evals[0] is λ_max).
        evals: np.ndarray shape (k,), float64.
        evecs: np.ndarray shape (n, k), columns are unit-norm eigenvectors
               (in the flat param space).
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

    _, n = _param_geometry(params)
    if k >= n:
        raise ValueError(f"ARPACK requires k < n; got k={k}, n={n}")

    # dtype: float32 for fp32 models (ARPACK runs in declared dtype)
    p_dtype = params[0].dtype
    dtype_np = np.float32 if p_dtype == torch.float32 else np.float64
    device = params[0].device

    # warm-start
    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(n).astype(dtype_np)
        v0 /= float(np.linalg.norm(v0))
    else:
        v0 = np.asarray(v0, dtype=dtype_np)

    n_hvp = {"count": 0}  # closure counter for diagnostics

    def _matvec(v_np: np.ndarray) -> np.ndarray:
        v_np = np.ascontiguousarray(v_np, dtype=dtype_np)
        v_t = torch.from_numpy(v_np).to(device).type(p_dtype)
        Hv_flat = _hvp_flat_chunked(model, loss_fn, X, y, v_t, params, chunk_size)
        n_hvp["count"] += 1
        return Hv_flat.detach().to("cpu").double().numpy().astype(dtype_np)

    A = LinearOperator((n, n), matvec=_matvec, dtype=dtype_np)
    try:
        evals, evecs = eigsh(
            A, k=k, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv
        )
    except ArpackNoConvergence as exc:
        # partial results — return whatever ARPACK found (best-effort)
        evals = exc.eigenvalues
        evecs = exc.eigenvectors
        if evals is None or len(evals) == 0:
            raise
    # eigsh returns ASCENDING; flip to DESCENDING so evals[0] is λ_max
    order = np.argsort(-evals)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs
