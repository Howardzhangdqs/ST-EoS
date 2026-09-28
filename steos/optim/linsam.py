"""Full-rank linearized SAM: g <- g + (rho/||g||) * H_B g.

Review-round-3 control arm. SAM (Foret et al. 2021) perturbs to
theta + rho*g/||g|| and re-queries the gradient there; its first-order
Taylor expansion is

    g_SAM = grad L(theta + rho*g/||g||)  ~=  g + (rho/||g||) * H g ,

an ALL-DIRECTION curvature-amplifying correction: along each Hessian
eigenvector v_i the gradient component is rescaled by 1 + rho*lam_i/||g||.
The eigenspace-edit family of this project (rankkx amplify) applies the same
sign of correction on a rank-k top-Hessian subspace, so "SAM is the
full-rank endpoint of that family" is exact only to first order. This arm
applies the first-order update DIRECTLY -- one HVP per step, NO
perturbed-point re-query -- which separates the two candidate mechanisms for
SAM's lambda_max reduction:

  * if this arm reproduces SAM's reduction, the deciding variable is the
    rank / globality of the correcting action (the eigenspace family merely
    lacks rank);
  * if it does not, the operative mechanism is SAM's nonlinear
    perturb-and-requery (second order and beyond), not rank per se.

Faithfulness notes (matching steos/optim/sam.py exactly where first order
allows):
  * H is the Hessian of the CURRENT training minibatch loss in TRAIN mode
    (batch BN statistics) -- sam_step also forwards the perturbed point in
    train mode, so this is the faithful linearization of THIS codebase's SAM.
  * Exactly ONE forward pass per step (BN running stats updated once, same
    count as a plain SGD step -- unlike sam_step's two). The linSAM-vs-SGD
    contrast therefore differs ONLY by the gradient correction.
  * The base optimizer (SGD + momentum + weight decay) is identical across
    the sgd / sam / linsam arms; rho is shared.
"""

from __future__ import annotations

import math

import torch

from steos.sharpness.hessian import _param_list


class LinSAMState:
    """Per-step diagnostics for the linsam arm (jsonl-logged every step)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.gnorm = float("nan")      # ||g||
        self.c = float("nan")          # correction scale rho/||g||
        self.hg_norm = float("nan")    # ||H g||
        self.corr_norm = float("nan")  # ||c * H g||  (correction magnitude)
        self.cos_g_hg = float("nan")   # cos(g, H g)
        self.lam_step = float("nan")   # g^T H g / ||g||^2  (per-step batch
                                       # sharpness along g -- free byproduct)
        self.degenerate = 0            # steps skipped for ||g|| ~ 0

    def metrics(self) -> dict:
        return dict(gnorm=self.gnorm, c=self.c, hg_norm=self.hg_norm,
                    corr_norm=self.corr_norm, cos_g_hg=self.cos_g_hg,
                    lam_step=self.lam_step, degenerate=self.degenerate)


def linsam_bnmatched_step(model, base_optimizer, loss_fn, X, y,
                          state: LinSAMState, rho: float):
    """linSAM with BatchNorm running-stat evolution matched to sam_step.

    Review-round-7 normalization control: sam_step performs TWO train-mode
    forwards per step (at θ and at θ + ρ·g/‖g‖), so BN running statistics
    update TWICE per step, while linsam_step's single forward updates them
    ONCE. Since the reference λ_max is measured in eval mode — where the
    forward USES the running statistics — the SAM-vs-linSAM contrast could
    in principle conflate the re-query with BN-state evolution. This arm
    removes that asymmetry: after computing the first-order corrected
    gradient (and BEFORE base.step()), it performs sam_step's second
    forward exactly — perturb θ by ε = ρ·g/‖g‖ (the RAW g, sam's own
    perturbation, NOT the corrected gradient), one no-grad train-mode
    forward on the SAME batch (running-stats update only), then restore θ.

    Consequences:
      * BN running-stat evolution is IDENTICAL to sam_step's (two updates,
        same batches, same perturbation points);
      * train-mode dynamics (batch statistics) are identical to linsam_step's;
      * the ONLY remaining difference vs sam_step is the gradient used for
        the step (first-order correction vs re-queried nonlinear gradient);
      * the ONLY difference vs linsam_step is the dummy second forward.

    Returns the pre-step minibatch loss (same convention as sam_step /
    linsam_step).
    """
    base_optimizer.zero_grad(set_to_none=True)
    params = _param_list(model)
    loss = loss_fn(model(X), y)                    # stats update 1 (= sam's fwd@θ)
    g = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = torch.cat([gi.reshape(-1) for gi in g])
    gg = float(g_flat.detach().pow(2).sum().item())
    if gg < 1e-24:
        state.degenerate += 1
        return float(loss.item())
    v = g_flat.detach()                            # raw g(θ) — sam's own ε basis
    gdotv = (g_flat * v).sum()                     # carries graph; d/dθ = H v = H g
    Hv = torch.autograd.grad(gdotv, params, retain_graph=False)
    hg_flat = torch.cat([h.reshape(-1) for h in Hv]).detach()

    gnorm = math.sqrt(gg)
    c = rho / gnorm
    gHg = float((v * hg_flat).sum().item())
    hgn = float(hg_flat.pow(2).sum().sqrt().item())
    state.gnorm = gnorm
    state.c = c
    state.hg_norm = hgn
    state.corr_norm = c * hgn
    state.cos_g_hg = gHg / max(gnorm * hgn, 1e-30)
    state.lam_step = gHg / gg

    new_flat = v + c * hg_flat                     # g + (rho/||g||) H g
    s = 0
    with torch.no_grad():
        for p in params:
            e = s + p.numel()
            p.grad = new_flat[s:e].view(p.shape).clone()
            s = e
        # ---- sam_step's second forward, reproduced exactly ----
        # Scale computed with sam_step's OWN accumulation order (per-parameter
        # .sum().item() in python floats, same iteration order) so ε is
        # bit-identical and the dummy forward's buffer updates are too.
        s2 = 0.0
        for gi in g:
            s2 += float(gi.detach().pow(2).sum().item())
        scale = rho / (s2 ** 0.5 + 1e-12)
        orig = {n: p.detach().clone() for n, p in model.named_parameters()}
        o = 0
        for gi in g:                                 # perturb by RAW g (not p.grad)
            params[o].add_(gi.detach() * scale)      # ASCENT, = sam_step's ε
            o += 1
        _ = loss_fn(model(X), y)                     # stats update 2, no autograd
        for n, p in model.named_parameters():        # restore θ
            p.copy_(orig[n])
    base_optimizer.step()
    return float(loss.item())


def linsam_step(model, base_optimizer, loss_fn, X, y, state: LinSAMState,
                rho: float):
    """One linearized-SAM step: p.grad <- g + (rho/||g||)*H_B g; base.step().

    One forward + one double-backward HVP with v = g (Pearlmutter; the graph
    from the single forward is reused, so no second training forward).
    Returns the pre-step minibatch loss (same convention as sam_step).
    """
    base_optimizer.zero_grad(set_to_none=True)
    params = _param_list(model)
    loss = loss_fn(model(X), y)
    g = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = torch.cat([gi.reshape(-1) for gi in g])
    gg = float(g_flat.detach().pow(2).sum().item())
    if gg < 1e-24:
        state.degenerate += 1
        return float(loss.item())
    v = g_flat.detach()                    # fixed vector = g(theta)
    gdotv = (g_flat * v).sum()             # carries graph; d/dtheta = H v = H g
    Hv = torch.autograd.grad(gdotv, params, retain_graph=False)
    hg_flat = torch.cat([h.reshape(-1) for h in Hv]).detach()

    gnorm = math.sqrt(gg)
    c = rho / gnorm
    gHg = float((v * hg_flat).sum().item())
    hgn = float(hg_flat.pow(2).sum().sqrt().item())
    state.gnorm = gnorm
    state.c = c
    state.hg_norm = hgn
    state.corr_norm = c * hgn
    state.cos_g_hg = gHg / max(gnorm * hgn, 1e-30)
    state.lam_step = gHg / gg

    new_flat = v + c * hg_flat            # g + (rho/||g||) H g
    s = 0
    with torch.no_grad():
        for p in params:
            e = s + p.numel()
            p.grad = new_flat[s:e].view(p.shape).clone()
            s = e
    base_optimizer.step()
    return float(loss.item())


def linsam_step_chunked(model, base_optimizer, loss_fn, X, y,
                        state: LinSAMState, rho: float, chunk: int):
    """linsam_step with the HVP double-backward chunked over the batch.

    Memory variant of linsam_step for large training batches (review-round
    10c): the create_graph/double-backward graph is restricted to `chunk`
    samples at a time, so peak activation memory is that of a chunk-sized
    linsam step regardless of the training batch size. With a mean-reduction
    loss, L = sum_c w_c L_c (w_c = |c|/N), so g = sum_c w_c g_c and
    H g = sum_c w_c H_c v with v = g -- each chunk term needs create_graph
    only on chunk c's graph.

    Semantics identical to linsam_step: same update p.grad = g +
    (rho/||g||) H g, same state metrics, same returned loss. Costs one
    extra full no-graph forward (pass A) plus the chunk loop (pass B),
    i.e. 1+C train-mode forwards per step: INTENDED FOR LAYERNORM MODELS
    (train-mode forwards have no running-stat side effects); do NOT use
    with BatchNorm. Float-associativity differs from the single-graph
    path (agreement verified on a unit test and on step-0 quantities,
    ~1e-6 relative); the legacy path remains the default.
    """
    base_optimizer.zero_grad(set_to_none=True)
    params = _param_list(model)
    n = X.shape[0]
    # ---- pass A: full-batch loss + v = g (no create_graph) ----
    loss = loss_fn(model(X), y)
    g = torch.autograd.grad(loss, params)
    g_flat = torch.cat([gi.reshape(-1) for gi in g]).detach()
    gg = float(g_flat.pow(2).sum().item())
    if gg < 1e-24:
        state.degenerate += 1
        return float(loss.item())
    v = g_flat
    # ---- pass B: chunked double-backward, Hv = sum_c w_c H_c v ----
    hv_flat = torch.zeros_like(v)
    for c0 in range(0, n, chunk):
        Xc = X[c0:c0 + chunk]
        yc = y[c0:c0 + chunk]
        w = Xc.shape[0] / n
        loss_c = loss_fn(model(Xc), yc)
        g_c = torch.autograd.grad(loss_c, params, create_graph=True)
        gc_flat = torch.cat([gi.reshape(-1) for gi in g_c])
        gdotv = (gc_flat * v).sum()          # carries chunk-c graph only
        Hv_c = torch.autograd.grad(gdotv, params, retain_graph=False)
        with torch.no_grad():
            s = 0
            for h in Hv_c:
                e = s + h.numel()
                hv_flat[s:e] += w * h.reshape(-1)
                s = e
    gnorm = math.sqrt(gg)
    c = rho / gnorm
    gHg = float((v * hv_flat).sum().item())
    hgn = float(hv_flat.pow(2).sum().sqrt().item())
    state.gnorm = gnorm
    state.c = c
    state.hg_norm = hgn
    state.corr_norm = c * hgn
    state.cos_g_hg = gHg / max(gnorm * hgn, 1e-30)
    state.lam_step = gHg / gg

    new_flat = v + c * hv_flat            # g + (rho/||g||) H g
    s = 0
    with torch.no_grad():
        for p in params:
            e = s + p.numel()
            p.grad = new_flat[s:e].view(p.shape).clone()
            s = e
    base_optimizer.step()
    return float(loss.item())
