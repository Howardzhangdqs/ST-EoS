"""CIFAR-100 on standard ResNet-18 — EoS + control-experiment runner.

This is the FIRST experiment addressing reviewer concern #1 (experimental
scale): the paper's control experiments (C1 weight-norm cap, C3 rank-1 plant,
rank-k sweep, X-SAM) are almost entirely MLP-based. ResNet-18 on CIFAR-100 is a
standard CV benchmark that moves us from "MLP-only" to "real CNN on a standard
dataset", which is the minimum bar reviewers set.

We use a TRUE ResNet-18 (torchvision, ~11.2M params) with the CIFAR adaptation
(3x3 first conv, stride 1, no maxpool) — NOT the 0.17M toy PreAct-ResNet in
steos/models/resnet.py. This matters: the reviewer's ask is explicitly about
"ResNet-18", and a toy 0.17M net is not a defensible answer to a scale concern.

The runner supports these optimizers via --opt:
    sgd            plain mini-batch SGD + momentum (baseline)
    rank1          rank-1 plant-action preconditioner (the Claim-1 actuator test)
    sam            Sharpness-Aware Minimization (the global action that DOES escape)
    rankk          rank-k plant-action (for the k-invariance check on a real CNN)
    xsam_faithful  Duan et al. 2026 X-SAM, faithful (NO clamp; fixed α, sign-gated,
                   unconditional) — the variant that actually fires every step.
    xsam_clamped   our existing xsam.py clamp variant (control arm; expected inert).

--dataset {cifar10, cifar100} selects the dataset (default cifar100); the
default run-name is prefixed accordingly (cifar10_r18_{opt}_s{seed}).

--augment (reviewer item 3) enables the standard CIFAR train-time recipe —
replicate-pad-4 + per-sample random 32x32 crop + horizontal flip (p=0.5) —
applied per TRAINING batch at draw time on GPU (steos/data/cifar.py caches
raw normalized tensors, which must stay unaugmented). Augmentation RNG is a
dedicated torch.Generator seeded from --seed, so runs stay seed-reproducible
and decoupled from the global torch stream. Eval / sharpness / ARPACK
measurement batches are NEVER augmented. Default (flag absent) is
bit-identical to the original runner: the augment path is fully guarded
behind the flag.

Each run records per-snapshot: λ_max (warm-started power iter on a fixed eval
batch in EVAL mode → deterministic Hessian via BN running stats), the single-HVP
Rayleigh proxies, |cos| alignments, train/eval loss, eval accuracy, and the EoS
diagnostic λ_max vs 2/η (the central quantity of the paper).

Usage (smoke test):
    python scripts/experiments/train_cifar100_resnet18.py --opt sgd --steps 200 --seed 0 \
        --run-name cifar100_r18_sgd_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.data.cifar import load_cifar
from steos.optim import SGD, register_param_names
from steos.optim.sam import sam_step
from steos.optim.eos_precond_rankk import EoSPrecondRankKState, eos_precond_rankk_step
from steos.optim.xsam import XSAMState, xsam_step
from steos.optim.xsam_faithful import XSAMFaithfulState, xsam_faithful_step
from steos.optim.true_lmax_plant import TrueLmaxState, true_lmax_plant_step
from steos.optim.rankkx_plant import RankKXState, rankkx_plant_step
from steos.optim.linsam import (LinSAMState, linsam_step, linsam_bnmatched_step,
                                linsam_step_chunked)
from steos.optim.sam_interp import sam_interp_step
from steos.sharpness import max_eigenvalue, rayleigh_proxies
from steos.sharpness.hessian import (
    _param_list, _hvp_flat_chunked, max_eigenvalue_arpack,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# BN running-stat protocol per arm (review round 7): sam_step performs TWO
# train-mode forwards per step (θ and θ+ε) so running stats update TWICE;
# plain sgd / linsam update ONCE; the matched controls below pin this count.
_BN_PROTOCOL = {
    "sgd": "single-update (1 train-mode forward)",
    "sam": "double-update (sam_step: forwards at θ and θ+ε)",
    "linsam": "single-update (1 train-mode forward)",
    "sam_bnfreeze": "frozen (official SAM protocol: no running-stat updates during the step)",
    "sgd_bn2x": "double-update (dummy no-grad forward at θ+ε, gradient from θ)",
    "linsam_bnmatched": "double-update (dummy no-grad forward at θ+ε, gradient = first-order correction)",
    "sam_interp": "double-update (forwards at θ and θ+ε)",
    "linsam_neg": "single-update (1 train-mode forward)",
}


def _bn_buffers(model):
    """Clone all BatchNorm running buffers (mean/var/num_batches_tracked)."""
    return [(m, m.running_mean.detach().clone(), m.running_var.detach().clone(),
             None if m.num_batches_tracked is None
             else m.num_batches_tracked.detach().clone())
            for m in model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]


def _restore_bn_buffers(bufs):
    for m, rm, rv, nbt in bufs:
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        if nbt is not None:
            m.num_batches_tracked.copy_(nbt)


@torch.no_grad()
def _global_grad_norm(model) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum().item())
    return s ** 0.5


def build_resnet18(num_classes: int = 100) -> nn.Module:
    """Standard torchvision ResNet-18 adapted for CIFAR (32x32).

    CIFAR adaptation (the canonical recipe): replace the 7x7 stride-2 first conv
    with a 3x3 stride-1 conv and remove the maxpool, so feature maps stay at
    32x32 through stem and we keep the spatial resolution that ResNet-18 was
    designed around at ImageNet scale. This is the standard "ResNet-18 on CIFAR"
    setup used across the EoS / sharpness literature.
    """
    import torchvision.models as tvm

    net = tvm.resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


@torch.no_grad()
def eval_acc(model, Xv, yv, bs=500):
    model.eval()
    correct = 0
    for i in range(0, len(Xv), bs):
        pred = model(Xv[i:i + bs]).argmax(1)
        correct += int((pred == yv[i:i + bs]).sum())
    model.train()
    return correct / len(Xv)


_AUG_PAD = 4  # CIFAR standard: pad 4 px each side -> 40x40, random 32x32 crop


def augment_batch(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Standard CIFAR train-time augmentation ON A NORMALIZED FLOAT BATCH
    (B,3,32,32), on-GPU at batch draw time — the dataset cache (raw
    normalized tensors) is never touched. Per-sample, each independent:

      1. replicate-pad 4 px each side            -> (B,3,40,40)
      2. random 32x32 crop (offsets ~ U{0..8}^2)
      3. horizontal flip, p=0.5

    Deterministic given ``gen`` (dedicated torch.Generator). All ops are
    forward-only on a no-grad batch (pad/gather/where are outside the
    autograd graph), so they are safe under
    torch.use_deterministic_algorithms(True).
    """
    B, C, H, W = x.shape
    S = H + 2 * _AUG_PAD
    xp = F.pad(x, (_AUG_PAD,) * 4, mode="replicate")             # (B,C,S,S)
    off = torch.randint(0, 2 * _AUG_PAD + 1, (B, 2), generator=gen,
                        device=x.device)                          # (B,2) dy,dx
    rows = off[:, 0:1] + torch.arange(H, device=x.device)         # (B,H)
    cols = off[:, 1:2] + torch.arange(W, device=x.device)         # (B,W)
    flat = (rows[:, :, None] * S + cols[:, None, :]).reshape(B, 1, H * W)
    out = torch.gather(xp.reshape(B, C, S * S), 2,
                       flat.expand(B, C, H * W)).reshape(B, C, H, W)
    flip = torch.rand(B, generator=gen, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(dims=[3]), out)


_MEAS_SEEDS = (20240810, 20240811, 20240812, 20240813)  # fixed → comparable across opt runs
_RESTART_ITERS = 200  # per-restart cap (tol-based early stop makes late-training cheap)


def measure_curvature(model, opt, Xeval, yeval, n_iter, warm_v=None):
    """λ_max via DETERMINISTIC MULTI-RESTART power iteration + single-HVP
    Rayleigh proxies, in eval mode on the deterministic batch.

    Each call runs R=len(_MEAS_SEEDS) independent cold-start power iterations
    from FIXED-seed inits and takes the MAX λ_max (and its eigenvector). This is
    (i) deterministic — same model ⇒ same λ_max regardless of training-RNG
    state, so cross-optimizer trajectories are directly comparable; and
    (ii) robust — the max over restarts defeats the two single-start failure
    modes: warm-start chains that seed onto a subdominant eigenvector and stay
    biased low the whole run, and a single cold-start whose fixed init is
    momentarily misaligned with the top eigenvector. The ``warm_v`` and
    ``n_iter`` args are ignored (overridden by ``_RESTART_ITERS``)."""
    ce = lambda p, y: F.cross_entropy(p, y)
    params = _param_list(model)
    model.eval()
    try:
        best_lam, best_v, best_n = -1.0, None, 0
        for sd in _MEAS_SEEDS:
            gen = torch.Generator(device=Xeval.device).manual_seed(sd)
            v_init = [torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                      for p in params]
            vn = torch.sqrt(torch.stack([(vv * vv).sum() for vv in v_init]).sum()).clamp(min=1e-12)
            v_init = [vv / vn for vv in v_init]
            lam, v, n_hvp = max_eigenvalue(
                model, ce, Xeval, yeval, n_iter=_RESTART_ITERS, return_vec=True,
                return_iters=True, v_init=v_init,
            )
            if lam > best_lam:
                best_lam, best_v, best_n = lam, v, n_hvp
        rec = {"lambda_max": float(best_lam), "lambda_max_n_hvp": int(best_n)}
        rec.update({k: float(val) for k, val in
                    rayleigh_proxies(model, opt, ce, Xeval, yeval, best_v, params).items()})
    finally:
        model.train()
    return rec, best_v


# Default eval batch for ARPACK outcome measurement (matches P0: seed=123, B=512).
_ARP_EVAL_SEED = 123
_ARP_DEFAULT_CHUNK = 1024   # P0 established 2048 OOMs on 16 GB; 1024 is safe.


def _subspace_min_cos(Va: np.ndarray, Vb: np.ndarray) -> float:
    """Cosine of the LARGEST principal angle between two column-orthonormal
    bases (n×k each): the minimum singular value of VaᵀVb. 1 ⇒ identical
    subspace, 0 ⇒ some direction rotated fully out."""
    s = np.linalg.svd(Va.T @ Vb, compute_uv=False)
    return float(s.min())


def measure_curvature_arpack(model, Xeval, yeval, k=3, tol=1e-6,
                             chunk=_ARP_DEFAULT_CHUNK, ncv=None,
                             return_evecs=False):
    """λ_max via ARPACK (tol=1e-6, k=3) on the deterministic eval batch in EVAL
    mode with BN save/restore. The GROUND-TRUTH outcome measurement for the
    true-λ_max plant experiment (preregistration §3): every ~200 steps this
    produces the plant-λ_max value used for the escape / pin(a) / pin(b)
    classification. Also returns the Ritz residual of the top eigenpair (a
    staleness/quality check independent of the plant's sensing-time Ritz) and
    the single-HVP gradient Rayleigh quotient λ̂_g for the proxy comparator.

    P0 Task C showed the existing power-iter ``measure_curvature`` under-
    estimates λ_max by mean +45 %, worst +65 % — that is exactly the artifact
    that made the old C3 plant inert. This function replaces it for the
    true-λ_max runs.

    ``return_evecs=True`` additionally returns the (n,k) eigenvector matrix as
    a torch tensor on-device (top-k spectrum / eigenspace-overlap logging).
    """
    ce = lambda p, y: F.cross_entropy(p, y)
    params = _param_list(model)
    model.eval()
    saved_bn = [(m, m.running_mean.clone(), m.running_var.clone())
                for m in model.modules() if hasattr(m, "running_mean")]
    try:
        evals, evecs = max_eigenvalue_arpack(
            model, ce, Xeval, yeval, params, k=k, tol=tol, chunk_size=chunk,
            ncv=ncv)
        lam1 = float(evals[0])
        lam2 = float(evals[1]) if len(evals) > 1 else float("nan")
        # Ritz residual ‖Hv−λv‖/|λ| via ONE more HVP on the top eigenvector
        v_t = torch.from_numpy(evecs[:, 0]).to(params[0].device).type(params[0].dtype)
        Hv_flat = _hvp_flat_chunked(model, ce, Xeval, yeval, v_t, params, chunk)
        denom = max(abs(lam1) * float(v_t.norm().item()), 1e-30)
        ritz = float((Hv_flat - lam1 * v_t).norm().item() / denom)
        # λ̂_g (single-HVP Rayleigh on the gradient)
        out = model(Xeval)
        loss = ce(out, yeval)
        g = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
        g = [gi.detach() for gi in g]
        Hg_flat = _hvp_flat_chunked(
            model, ce, Xeval, yeval,
            torch.cat([gi.reshape(-1) for gi in g]), params, chunk)
        gg = float(sum((gi * gi).sum().item() for gi in g))
        gHg = float((torch.cat([gi.reshape(-1) for gi in g]) * Hg_flat).sum().item())
        lhg = gHg / max(gg, 1e-30)
        rec = {"lambda_max": lam1, "lambda_2": lam2,
               "lambda_max_arpack_ritz": ritz, "lambda_hat_grad": lhg}
        if return_evecs:
            rec["_evecs"] = evecs          # np.ndarray (n, k), columns descending
            rec["_evals"] = evals
    finally:
        for m, rm, rv in saved_bn:
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
        model.train()
    return rec


# Number of fresh minibatches averaged per measurement point for the E_B[·]
# batch-sharpness estimate (Andreyev & Beneventano's step sharpness).
_BS_SHARP_M = 4


def _grad_rayleigh_on_batch(model, X, y, chunk=None) -> float:
    """g_B^T H_B g_B / g_B^T g_B on batch (X, y) — the EoSS *batch sharpness*
    (expected directional curvature along the batch gradient), NOT the
    minibatch top eigenvalue. Eval mode with BN running stats saved/restored
    (same measurement convention as the ARPACK λ̂_g of measure_curvature_
    arpack). Cost: 1 forward for g + 1 Pearlmutter HVP.
    The gradient is accumulated CHUNK-WISE with create_graph=False (the HVP
    below treats g as a fixed vector, so the double-backward graph is
    unneeded): per-sample losses are independent in eval mode, so the
    chunk-weighted sum of chunk-mean gradients equals the full-batch
    gradient exactly, while peak memory stays bounded by the chunk size
    (the previous single-forward create_graph=True path OOM'd at bs=2048 on
    a 16 GB GPU). For batch <= chunk this is bitwise the legacy path."""
    ce_ = lambda p, yy: F.cross_entropy(p, yy)
    params = _param_list(model)
    model.eval()
    saved_bn = [(m, m.running_mean.clone(), m.running_var.clone())
                for m in model.modules() if hasattr(m, "running_mean")]
    try:
        n = X.shape[0]
        cs = chunk or n
        g_flat = None
        for i in range(0, n, cs):
            Xc, yc = X[i:i + cs], y[i:i + cs]
            nc = Xc.shape[0]
            loss = ce_(model(Xc), yc) * (nc / n)
            gs = torch.autograd.grad(loss, params)
            gf = torch.cat([gi.reshape(-1) for gi in gs])
            g_flat = gf if g_flat is None else g_flat + gf
        gg = float(g_flat.detach().pow(2).sum().item())
        Hg = _hvp_flat_chunked(model, ce_, X, y, g_flat.detach(), params, chunk)
        gHg = float((g_flat.detach() * Hg).sum().item())
        return gHg / max(gg, 1e-30)
    finally:
        for m, rm, rv in saved_bn:
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
        model.train()


def _taylor_residual_on_batch(model, X, y, rho, chunk=None) -> dict:
    """Nonlinear re-query content of SAM's perturbed-point evaluation
    (review round 5).  With g = ∇_θ L_B(θ), the SAM ascent perturbation
    ε = ρ·g/‖g‖, and the first-order (linSAM) prediction of the perturbed
    gradient g_lin = g + (ρ/‖g‖)·H_B g, returns on the batch (X, y):

        taylor_gnorm   ‖g‖
        taylor_d_total ‖∇L_B(θ+ε) − g‖ / ‖g‖     (total re-query displacement)
        taylor_d_lin   ‖g_lin − g‖ / ‖g‖         (its linearized part)
        taylor_res_rel ‖∇L_B(θ+ε) − g_lin‖ / ‖g‖ (the NONLINEAR residual —
                       zero iff SAM's re-query equals its first-order expansion)
        taylor_cos_pert_lin  cos(∇L_B(θ+ε), g_lin)

    TRAIN-mode BN batch statistics — the faithful semantics of sam_step's two
    forwards (see the faithfulness notes in steos/optim/linsam.py) — with
    running stats and θ saved/restored: a pure measurement with no effect on
    the training path.  Cost ≈ 3 forward + 3 backward passes on B=|X|."""
    ce_ = lambda p, yy: F.cross_entropy(p, yy)
    params = _param_list(model)
    was_training = model.training
    model.train()
    saved_bn = [(m, m.running_mean.clone(), m.running_var.clone())
                for m in model.modules() if hasattr(m, "running_mean")]
    saved_theta = [p.detach().clone() for p in params]
    out = {"taylor_gnorm": float("nan"), "taylor_d_total": float("nan"),
           "taylor_d_lin": float("nan"), "taylor_res_rel": float("nan"),
           "taylor_cos_pert_lin": float("nan")}
    try:
        loss = ce_(model(X), y)
        g = torch.autograd.grad(loss, params)
        g_flat = torch.cat([gi.reshape(-1) for gi in g]).detach()
        gn = float(g_flat.norm().item())
        out["taylor_gnorm"] = gn
        if gn < 1e-12:
            return out
        # linearized perturbed-point gradient (one Pearlmutter HVP; the HVP
        # path runs its own fresh forwards, so no create_graph needed here)
        Hg = _hvp_flat_chunked(model, ce_, X, y, g_flat, params, chunk)
        g_lin = g_flat + (rho / gn) * Hg
        out["taylor_d_lin"] = float((g_lin - g_flat).norm().item()) / gn
        # the actual re-query: perturb θ by ε = ρ·ĝ (ASCENT — sam_step's
        # sign), evaluate the gradient there, restore θ afterwards.
        eps = (rho / gn) * g_flat
        with torch.no_grad():
            s = 0
            for p in params:
                e = s + p.numel()
                p.add_(eps[s:e].view(p.shape))
                s = e
        loss2 = ce_(model(X), y)
        g2 = torch.autograd.grad(loss2, params)
        g2_flat = torch.cat([gi.reshape(-1) for gi in g2]).detach()
        out["taylor_d_total"] = float((g2_flat - g_flat).norm().item()) / gn
        out["taylor_res_rel"] = float((g2_flat - g_lin).norm().item()) / gn
        denom = float(g2_flat.norm().item()) * float(g_lin.norm().item())
        out["taylor_cos_pert_lin"] = float((g2_flat @ g_lin).item()) / max(denom, 1e-30)
        return out
    finally:
        with torch.no_grad():
            for p, sp in zip(params, saved_theta):
                p.copy_(sp)
        for m, rm, rv in saved_bn:
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
        if not was_training:
            model.eval()


def _linsam_neg_step(model, base_optimizer, loss_fn, X, y, state: LinSAMState,
                     rho: float, sign: float = -1.0, chunk: int = 0):
    """linSAM with a SIGNED first-order correction (review round 11).

    Update: g + sign·(ρ/‖g‖)·H_B g  —  sign=+1 is exactly linsam_step,
    sign=−1 is the norm-matched full-rank negative-Hg damping control
    (ICLR reviewer ask): g − (ρ/‖g‖)·H_B g, testing whether the stochastic
    null of the eigenspace arms is a sign-rule failure vs weak low-rank
    actuation.

    NORM-MATCHED BY CONSTRUCTION: the sign flip preserves the correction
    norm, ‖sign·(ρ/‖g‖)H_B g‖ = (ρ/‖g‖)·‖H_B g‖ — identical to linSAM's,
    so any behavioral difference is attributable to the sign alone.

    Implementation (zero changes to steos/optim/linsam.py): the sign is
    threaded through the sign of ρ in the UNMODIFIED steos linsam code
    path — c = ρ/‖g‖ is the ONLY ρ-dependent quantity in linsam_step /
    linsam_step_chunked, so ρ→sign·ρ yields precisely the signed update.
    The negative arm therefore shares the identical HVP machinery (legacy
    single-graph path for chunk=0, chunked double-backward for chunk>0),
    BN single-update protocol, LinSAMState diagnostics (state.c/corr_norm
    carry the sign; |corr_norm| is the correction norm), RNG stream and
    returned loss. Same caveat as --linsam-chunk: chunk>0 performs 1+C
    train-mode forwards per step — LayerNorm models only, never BatchNorm.
    """
    eff_rho = sign * rho
    if chunk > 0:
        return linsam_step_chunked(model, base_optimizer, loss_fn, X, y,
                                   state, rho=eff_rho, chunk=chunk)
    return linsam_step(model, base_optimizer, loss_fn, X, y, state,
                       rho=eff_rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["sgd", "rank1", "rankk", "sam",
                                      "xsam_faithful", "xsam_clamped", "true_lmax",
                                      "amplify", "rankkx", "linsam",
                                      # review-round-7 BN-protocol controls:
                                      "sam_bnfreeze", "sgd_bn2x", "linsam_bnmatched",
                                      # review-round-9 τ-interpolation:
                                      "sam_interp",
                                      # review-round-11 norm-matched
                                      # negative-Hg damping control:
                                      "linsam_neg"],
                    default="sgd")
    ap.add_argument("--model", choices=["resnet18", "vit_tiny"], default="resnet18",
                    help="backbone: resnet18 (BatchNorm, canonical) or "
                         "vit_tiny (timm vit_tiny_patch16_224 reconfigured to "
                         "32x32/patch4 — LayerNorm throughout, no running "
                         "stats; SDPA forced to the math backend for HVPs)")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--sharp-every", type=int, default=50)
    ap.add_argument("--sharp-batch", type=int, default=256)
    ap.add_argument("--sharp-niter", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    # optimizer-specific knobs
    ap.add_argument("--rho", type=float, default=0.05, help="SAM perturbation radius")
    ap.add_argument("--linsam-chunk", type=int, default=0,
                    help="linsam: chunk the training-batch HVP double-backward "
                         "to this many samples per chunk (memory cap for large "
                         "training batches; 0 = legacy single-graph path). "
                         "LayerNorm models only: the chunked path performs "
                         "1+C train-mode forwards per step, which would "
                         "double-update BatchNorm running stats.")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="sam_interp: blend τ in g = (1−τ)·g_lin + τ·g_SAM "
                         "(τ=1 recovers SAM exactly, τ=0 the linearized "
                         "BN-matched update)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="X-SAM-faithful fixed correction strength (Duan et al. α)")
    ap.add_argument("--k", type=int, default=1, help="rank for rankk/rank1")
    ap.add_argument("--n-sweeps", type=int, default=3, help="power-iter sweeps for rankk")
    ap.add_argument("--no-clamp", action="store_true",
                    help="clamp-free C3 plant (bridge experiment): rank-1/rank-k fires "
                         "every step with no max(·,1) gate; appends _noclamp to run-name")
    ap.add_argument("--deterministic", action="store_true",
                    help="enable deterministic CUDA (reproducible runs; clean cross-opt comparison)")
    ap.add_argument("--augment", action="store_true",
                    help="standard CIFAR train-time augmentation (replicate-pad-4 + "
                         "random 32x32 crop + horizontal flip), per training batch at "
                         "draw time; measurement batches stay UN-augmented. Off by "
                         "default (bit-identical legacy path).")
    ap.add_argument("--save-final", action="store_true",
                    help="persist final weights to <run>/final_model.pt after "
                         "training (state_dict incl. BN running buffers) — the "
                         "200epf / r11 checkpoint protocol for offline analyses. "
                         "Gated as of review round 11; training dynamics, RNG "
                         "streams and metrics.csv columns are unaffected (the "
                         "save happens after manifest write, consumes no RNG).")
    # --- true-λ_max plant (ST-EoS-PR-01) ---
    ap.add_argument("--plant-N", type=int, default=20,
                    help="true_lmax: ARPACK sensing refresh period (steps). "
                         "Preregistration default 20.")
    ap.add_argument("--lam-targ", type=float, default=None,
                    help="true_lmax: λ_targ for divisor (=damping threshold). "
                         "Default 2/lr (Preregistration: 2/0.1 = 20).")
    ap.add_argument("--plant-tol", type=float, default=1e-2,
                    help="true_lmax: ARPACK tol at sensing (Preregistration 1e-2).")
    ap.add_argument("--plant-k", type=int, default=2,
                    help="true_lmax: ARPACK k at sensing (Preregistration 2).")
    ap.add_argument("--amplify", type=float, default=0.5,
                    help="amplify: FIXED curvature-amplifying correction strength "
                         "α in g_eff = g + α·(g·v̂_top)·v̂_top (γ = −α < 0; the "
                         "sign-flip control of the 2-D sign map). Sensing is the "
                         "true_lmax ARPACK machinery.")
    # --- rank-k ARPACK-sensed plant (review-round-2 rank interpolation) ---
    ap.add_argument("--edit", choices=["damp", "amplify"], default="damp",
                    help="rankkx: edit sign. damp = divisor damping per sensed "
                         "direction (true_lmax semantics, k>1); amplify = fixed "
                         "g + α·Σᵢ(g·v̂ᵢ)v̂ᵢ (sign-flip control, k>1).")
    ap.add_argument("--sense-current", action="store_true",
                    help="rankkx: sense on the CURRENT training minibatch instead "
                         "of the fixed eval batch (fixed-batch-confound control).")
    ap.add_argument("--schedule", choices=["none", "cosine"], default="none",
                    help="learning-rate schedule (cosine for the standard long "
                         "recipe; applied per-step to the base optimizer).")
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="cosine schedule floor (final lr = --lr-min).")
    ap.add_argument("--spectrum-k", type=int, default=0,
                    help=">0: at each outcome measurement ALSO run ARPACK k=k "
                         "(tol 1e-4) and log λ1..λk, controller-direction overlaps "
                         "and consecutive-subspace turnover (spectral-"
                         "redistribution diagnostic).")
    ap.add_argument("--multibatch", action="store_true",
                    help="cross-batch sensing study (sgd): at each outcome "
                         "measurement, additionally measure ARPACK k=3 on two "
                         "more fixed batches (seeds 124/125) and on the CURRENT "
                         "training minibatch; log λ and top-1/top-3 eigenspace "
                         "overlaps vs the reference batch.")
    ap.add_argument("--eval-batch", type=int, default=512,
                    help="deterministic eval batch for sensing + outcome (P0: 512).")
    ap.add_argument("--chunk-size", type=int, default=_ARP_DEFAULT_CHUNK,
                    help="chunked-HVP batch size (memory cap). 1024 fits 16 GB GPU.")
    ap.add_argument("--outcome-arpack", action="store_true",
                    help="use ARPACK (k=3, tol=1e-6) instead of multi-restart power "
                         "iter for the every-N outcome λ_max measurement. Required "
                         "for the Preregistration plant + vanilla reference runs.")
    ap.add_argument("--outcome-every", type=int, default=200,
                    help="ARPACK outcome measurement period (Preregistration 200).")
    ap.add_argument("--outcome-tol", type=float, default=1e-6,
                    help="ARPACK tol at outcome measurement (Preregistration 1e-6).")
    ap.add_argument("--outcome-k", type=int, default=3,
                    help="ARPACK k at outcome measurement (Preregistration 3).")
    ap.add_argument("--outcome-ncv", type=int, default=None,
                    help="ARPACK ncv (None = ARPACK default 20). Benchmarked on "
                         "ResNet-18/CIFAR-10 B=512: k=1/ncv=8 needs 9 matvecs "
                         "vs 21 at defaults (~2.6x cheaper, rel err ≤1e-7).")
    args = ap.parse_args()

    if args.deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LR, MOM, BS = args.lr, args.mom, args.bs
    STEPS, SHARP_EVERY, SHARP_BATCH, SHARP_NITER, SEED = (
        args.steps, args.sharp_every, args.sharp_batch, args.sharp_niter, args.seed)
    n_classes = 10 if args.dataset == "cifar10" else 100
    dtag = "cifar10" if args.dataset == "cifar10" else "cifar100"
    noclamp_tag = "_noclamp" if args.no_clamp else ""
    mtag = "r18" if args.model == "resnet18" else "vittiny"
    run_name = args.run_name or f"{dtag}_{mtag}_{args.opt}{noclamp_tag}_s{SEED}"
    RUN_DIR = Path("runs") / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading CIFAR-{n_classes} ...", flush=True)
    Xtr, ytr, task, Xte, yte = load_cifar(n_classes=n_classes, device=DEV)
    N = len(Xtr)
    rng_eval = np.random.default_rng(123)
    idx_eval = rng_eval.integers(0, N, SHARP_BATCH)
    Xeval, yeval = Xtr[idx_eval], ytr[idx_eval]
    # ARPACK eval batch — deterministic B=512 subset, same seed for ALL runs
    # (plant + vanilla reference) so λ_max values are directly comparable.
    # Matches P0 diagnostics exactly.
    rng_eval_arm = np.random.default_rng(_ARP_EVAL_SEED)
    idx_eval_arm = rng_eval_arm.integers(0, N, args.eval_batch)
    Xeval_arm, yeval_arm = Xtr[idx_eval_arm], ytr[idx_eval_arm]

    if args.model == "vit_tiny":
        from steos.models import build_model
        model = build_model("timm_vit_tiny", num_classes=n_classes).to(DEV)
        # SDPA math backend GLOBALLY: the flash/efficient/cudnn attention
        # kernels lack the double-backward derivative needed by every HVP
        # path in this script (ARPACK sensing, linSAM HVP, batch sharpness,
        # Taylor residual). Same requirement as train_timm.py's
        # measure_sharpness, applied run-wide so no call site is missed.
        import torch.backends.cuda as _bc
        _bc.enable_math_sdp(True)
        _bc.enable_flash_sdp(False)
        _bc.enable_mem_efficient_sdp(False)
        _bc.enable_cudnn_sdp(False)
    else:
        model = build_resnet18(num_classes=n_classes).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params/1e6:.2f}M params | opt={args.opt} | dataset={args.dataset} | "
          f"lr={LR} (2/η={2/LR:g}) | bs={BS} | steps={STEPS} | "
          f"augment={bool(args.augment)}"
          + (f" | BN protocol: {_BN_PROTOCOL.get(args.opt, 'n/a (no BN-relevant arm)')}"
             if args.model == "resnet18" else " | normalization: LayerNorm (no running stats)"),
          flush=True)

    ce = lambda p, y: F.cross_entropy(p, y)
    base = SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=args.wd)
    register_param_names(base, model)
    rank_state = None
    if args.opt in ("rank1", "rankk"):
        rank_state = EoSPrecondRankKState(k=args.k)
    xsam_state = None
    if args.opt == "xsam_faithful":
        xsam_state = XSAMFaithfulState()
    elif args.opt == "xsam_clamped":
        xsam_state = XSAMState()
    # true-λ_max plant (Preregistration ST-EoS-PR-01) + its amplify variant
    plant_state = None
    if args.opt in ("true_lmax", "amplify"):
        lam_targ = args.lam_targ if args.lam_targ is not None else 2.0 / LR
        plant_state = TrueLmaxState(
            refresh_n=args.plant_N, lam_targ=lam_targ,
            chunk_size=args.chunk_size, sense_tol=args.plant_tol, sense_k=args.plant_k,
            amplify=(args.amplify if args.opt == "amplify" else None))
        print(f"[{args.opt}] λ_targ={lam_targ:.3f}  refresh_N={args.plant_N}  "
              f"sense tol={args.plant_tol} k={args.plant_k}  "
              f"eval-batch={args.eval_batch}  chunk={args.chunk_size}"
              + (f"  AMPLIFY α={args.amplify}" if args.opt == "amplify" else ""),
              flush=True)
    # rank-k ARPACK-sensed plant (review round 2: the rank interpolation)
    rankkx_state = None
    if args.opt == "rankkx":
        lam_targ = args.lam_targ if args.lam_targ is not None else 2.0 / LR
        rankkx_state = RankKXState(
            k=args.k, refresh_n=args.plant_N, lam_targ=lam_targ,
            chunk_size=args.chunk_size, sense_tol=args.plant_tol,
            edit=args.edit, amplify=args.amplify, sense_current=args.sense_current)
        print(f"[rankkx] edit={args.edit} k={args.k} λ_targ={lam_targ:.3f}  "
              f"refresh_N={args.plant_N} sense tol={args.plant_tol}  "
              f"sense_batch={'CURRENT-minibatch' if args.sense_current else 'fixed-eval'}  "
              f"eval-batch={args.eval_batch}  chunk={args.chunk_size}"
              + (f"  AMPLIFY α={args.amplify}" if args.edit == "amplify" else ""),
              flush=True)

    # full-rank linearized-SAM arm (review round 3: rank-vs-requery control)
    # + BN-matched variant (review round 7: dummy second forward, see
    #   linsam_bnmatched_step — isolates the re-query from BN-state evolution)
    linsam_state = None
    if args.opt in ("linsam", "linsam_bnmatched"):
        linsam_state = LinSAMState()
        print(f"[{args.opt}] first-order SAM match: g ← g + (ρ/‖g‖)·H_B g  "
              f"ρ={args.rho}  (no perturbed-point re-query"
              + ("; BN stats double-updated to match sam_step)" if args.opt == "linsam_bnmatched"
                 else ")"), flush=True)
        if args.opt == "linsam" and args.linsam_chunk > 0:
            print(f"[linsam] chunked HVP: {args.linsam_chunk} samples/chunk on "
                  f"bs {args.bs} ({-(-args.bs // args.linsam_chunk)} chunks; "
                  f"LayerNorm-only path)", flush=True)
    elif args.opt == "linsam_neg":
        # review round 11: norm-matched negative-Hg damping control (see
        # _linsam_neg_step) — same defaults/protocol as linsam, sign flipped.
        linsam_state = LinSAMState()
        print(f"[linsam_neg] negative-sign linSAM control: g ← g − (ρ/‖g‖)·H_B g  "
              f"ρ={args.rho}  (correction norm identical to linsam by "
              f"construction; sign-rule vs low-rank-actuation adjudication; "
              f"no perturbed-point re-query)", flush=True)
        if args.linsam_chunk > 0:
            print(f"[linsam_neg] chunked HVP: {args.linsam_chunk} samples/chunk on "
                  f"bs {args.bs} ({-(-args.bs // args.linsam_chunk)} chunks; "
                  f"LayerNorm-only path)", flush=True)
    elif args.opt == "sam_interp":
        linsam_state = LinSAMState()
        print(f"[sam_interp] τ-blend: g ← (1−τ)·g_lin + τ·g_SAM  ρ={args.rho}  "
              f"τ={args.tau}  (two forwards: BN double-update, sam's ε; "
              f"τ=1 ≡ sam, τ=0 ≡ linsam_bnmatched)", flush=True)

    rows = []
    plant_jsonl = None
    if plant_state is not None or rankkx_state is not None or linsam_state is not None:
        plant_jsonl = open(RUN_DIR / "plant_per_step.jsonl", "w", buffering=1)
    rng = np.random.default_rng(SEED)
    # Dedicated RNG for the E_B[·] batch-sharpness minibatch draws — NEVER the
    # training stream rng, so trajectories stay bit-identical to legacy runs.
    rng_bs = np.random.default_rng(SEED + 977)
    # --multibatch: two extra fixed eval batches (seeds 124/125) for the
    # cross-batch sensing study — same size as the reference batch.
    X_mb_extra = []
    if args.multibatch:
        for sd in (124, 125):
            r = np.random.default_rng(sd)
            idx = r.integers(0, N, args.eval_batch)
            X_mb_extra.append((sd, Xtr[idx], ytr[idx]))
    # --augment: dedicated augmentation RNG stream, seeded from --seed (+1
    # decouples it from the global torch stream). Created ONLY under the
    # flag, so the flag-off path is the exact legacy code path.
    aug_gen = torch.Generator(device=DEV).manual_seed(SEED + 1) if args.augment else None
    warm_v = None
    warm_evecs = None   # previous measurement's eigenvector matrix (turnover)
    t0 = time.time()
    n_fired_cum = 0

    for step in range(STEPS):
        if args.schedule == "cosine":
            base.param_groups[0]["lr"] = args.lr_min + 0.5 * (LR - args.lr_min) * (
                1.0 + np.cos(np.pi * step / max(STEPS - 1, 1)))
        idx = rng.integers(0, N, BS)
        Xb, yb = Xtr[idx], ytr[idx]
        if aug_gen is not None:
            # augment TRAIN batches only — Xeval / Xeval_arm / Xte used by
            # the measurement paths below are never passed through here
            Xb = augment_batch(Xb, aug_gen)

        if args.opt == "sgd":
            base.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xb), yb)
            loss.backward()
            base.step()
            loss_val = float(loss.item())
        elif args.opt == "sam":
            loss_val = sam_step(model, base, ce, Xb, yb, rho=args.rho)
        elif args.opt == "sam_bnfreeze":
            # Official SAM BatchNorm protocol (disable_running_stats): freeze
            # running stats across the WHOLE step — both train-mode forwards
            # still use batch statistics (training dynamics identical to the
            # sam arm); only the eval-time normalization state stops evolving.
            bnbuf = _bn_buffers(model)
            loss_val = sam_step(model, base, ce, Xb, yb, rho=args.rho)
            _restore_bn_buffers(bnbuf)
        elif args.opt == "sgd_bn2x":
            # Plain-SGD update + sam_step's SECOND train-mode forward
            # reproduced exactly (θ+ρ·g/‖g‖, same batch, no autograd): BN
            # running stats evolve as in the sam arm while the update stays
            # pure SGD — isolates the BN-update-count effect on the
            # eval-mode λ_max measurement.
            base.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xb), yb)
            loss.backward()
            gnorm2 = _global_grad_norm(model)
            if gnorm2 > 1e-12:
                scale2 = args.rho / (gnorm2 + 1e-12)
                orig2 = {n: p.detach().clone() for n, p in model.named_parameters()}
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.add_(p.grad * scale2)     # ASCEND, = sam_step's ε
                    _ = ce(model(Xb), yb)               # stats update 2 only
                    for n, p in model.named_parameters():
                        p.copy_(orig2[n])
            base.step()
            loss_val = float(loss.item())
        elif args.opt in ("rank1", "rankk"):
            loss_val = eos_precond_rankk_step(
                model, base, ce, Xb, yb, rank_state,
                lr=LR, k=args.k, n_sweeps=args.n_sweeps, clamp=not args.no_clamp)
            n_fired_cum += rank_state.n_fired
        elif args.opt == "xsam_faithful":
            loss_val = xsam_faithful_step(
                model, base, ce, Xb, yb, xsam_state,
                lr=LR, rho=args.rho, alpha=args.alpha)
            n_fired_cum += int(xsam_state.fired)  # always 1 (unconditional correction)
        elif args.opt == "xsam_clamped":
            loss_val = xsam_step(
                model, base, ce, Xb, yb, xsam_state,
                lr=LR, rho=args.rho, use_sam=True)
            n_fired_cum += int(xsam_state.fired)  # 0/1 depending on the clamp
        elif args.opt in ("true_lmax", "amplify"):
            loss_val = true_lmax_plant_step(
                model, base, ce, Xb, yb, Xeval_arm, yeval_arm, plant_state)
            n_fired_cum += int(plant_state.fire)
            # per-step plant log (Preregistration §3: λ_sensed, divisor, damping,
            # is_refresh, fire, ritz)
            if plant_jsonl is not None:
                plant_jsonl.write(json.dumps(plant_state.metrics()) + "\n")
        elif args.opt == "rankkx":
            loss_val = rankkx_plant_step(
                model, base, ce, Xb, yb, Xeval_arm, yeval_arm, rankkx_state)
            n_fired_cum += rankkx_state.n_fired
            if plant_jsonl is not None:
                plant_jsonl.write(json.dumps(rankkx_state.metrics()) + "\n")
        elif args.opt in ("linsam", "linsam_bnmatched", "linsam_neg"):
            if args.opt == "linsam_neg":
                # r11 negative-sign control: same HVP path / state metrics /
                # BN protocol as linsam, sign threaded via _linsam_neg_step.
                loss_val = _linsam_neg_step(
                    model, base, ce, Xb, yb, linsam_state,
                    rho=args.rho, sign=-1.0, chunk=args.linsam_chunk)
            elif args.opt == "linsam" and args.linsam_chunk > 0:
                loss_val = linsam_step_chunked(
                    model, base, ce, Xb, yb, linsam_state, rho=args.rho,
                    chunk=args.linsam_chunk)
            else:
                step_fn = linsam_bnmatched_step if args.opt == "linsam_bnmatched" else linsam_step
                loss_val = step_fn(
                    model, base, ce, Xb, yb, linsam_state, rho=args.rho)
            if plant_jsonl is not None:
                plant_jsonl.write(json.dumps(linsam_state.metrics()) + "\n")
        elif args.opt == "sam_interp":
            loss_val = sam_interp_step(
                model, ce, Xb, yb, base, rho=args.rho, tau=args.tau,
                state=linsam_state)
            if plant_jsonl is not None:
                plant_jsonl.write(json.dumps(linsam_state.metrics()) + "\n")

        # Outcome measurement (every --outcome-every for ARPACK mode, else every
        # --sharp-every for the legacy power-iter path).
        do_outcome_arpack = (
            args.outcome_arpack and (step % args.outcome_every == 0 or step == STEPS - 1))
        do_outcome_legacy = (
            not args.outcome_arpack and (step % SHARP_EVERY == 0 or step == STEPS - 1))
        if do_outcome_arpack:
            curv = measure_curvature_arpack(
                model, Xeval_arm, yeval_arm,
                k=max(args.outcome_k, args.spectrum_k), tol=args.outcome_tol,
                chunk=args.chunk_size, ncv=args.outcome_ncv,
                return_evecs=(args.spectrum_k > 0 or args.multibatch))
            warm_v = None  # not used in ARPACK mode
            with torch.no_grad():
                tr_loss = float(F.cross_entropy(model(Xeval_arm), yeval_arm))
                acc = eval_acc(model, Xte[:2000], yte[:2000])
            if plant_state is not None:
                fired_frac = n_fired_cum / max(1, step + 1)
            elif rankkx_state is not None:
                fired_frac = n_fired_cum / max(1, (step + 1) * rankkx_state.k)
            elif rank_state:
                fired_frac = n_fired_cum / max(1, (step + 1) * args.k)
            elif xsam_state is not None:
                fired_frac = n_fired_cum / max(1, step + 1)
            else:
                fired_frac = float("nan")
            # ---- spectral-redistribution + cross-batch diagnostics (review r2)
            spec_row = {}
            if args.spectrum_k > 0 and "_evecs" in curv:
                evals_sp, evecs_sp = curv["_evals"], curv["_evecs"]
                for i in range(min(args.spectrum_k, len(evals_sp))):
                    spec_row[f"lambda_spec{i + 1}"] = float(evals_sp[i])
                if rankkx_state is not None and rankkx_state.V_flat is not None:
                    # controller's sensed directions vs the OUTCOME-measured
                    # eigenvectors (fresh, tol 1e-6, same batch): does the
                    # damped/amplified subspace still span the top of the
                    # spectrum, or has another direction taken over λ_max?
                    Vc = torch.stack(rankkx_state.V_flat, dim=1)          # (n, k) gpu
                    Vo = torch.from_numpy(evecs_sp).to(Vc.device, Vc.dtype)
                    M = (Vc.T @ Vo).abs().cpu().numpy()                    # (k, k)
                    j = int(np.argmax(M[0]))
                    spec_row["cos_ctrl1_v1"] = float(M[0, 0])
                    spec_row["ctrl1_top_out_index"] = j + 1
                    spec_row["ctrl1_top_out_cos"] = float(M[0, j])
                    kk = min(rankkx_state.k, M.shape[1])
                    spec_row["ctrlk_outk_min_cos"] = _subspace_min_cos(
                        Vc.cpu().numpy()[:, :kk], evecs_sp[:, :kk])
                if warm_evecs is not None:
                    kk = min(args.spectrum_k, evecs_sp.shape[1], warm_evecs.shape[1])
                    spec_row["spec_turnover_min_cos"] = _subspace_min_cos(
                        evecs_sp[:, :kk], warm_evecs[:, :kk])
                    spec_row["cos_v1_prev"] = float(
                        abs(evecs_sp[:, 0] @ warm_evecs[:, 0]))
                warm_evecs = evecs_sp
            if args.multibatch and "_evecs" in curv:
                ref_ev = curv["_evecs"]
                for sd, Xm, ym in X_mb_extra:
                    r = measure_curvature_arpack(
                        model, Xm, ym, k=3, tol=1e-6, chunk=args.chunk_size,
                        return_evecs=True)
                    spec_row[f"lam1_b{sd}"] = r["lambda_max"]
                    spec_row[f"cos_v1_ref_b{sd}"] = float(
                        abs(ref_ev[:, 0] @ r["_evecs"][:, 0]))
                    spec_row[f"sub_min_ref_b{sd}"] = _subspace_min_cos(
                        ref_ev, r["_evecs"])
                # the CURRENT training minibatch (the batch this step trained
                # on, as-is): the "is the fixed-batch v̂ the wrong direction?"
                # confound measurement
                r = measure_curvature_arpack(
                    model, Xb, yb, k=3, tol=1e-6, chunk=args.chunk_size,
                    return_evecs=True)
                spec_row["lam1_curmb"] = r["lambda_max"]
                spec_row["cos_v1_ref_curmb"] = float(
                    abs(ref_ev[:, 0] @ r["_evecs"][:, 0]))
                spec_row["sub_min_ref_curmb"] = _subspace_min_cos(
                    ref_ev, r["_evecs"])
                # Batch sharpness proper (Andreyev & Beneventano EoSS):
                # λ_step = E_B[g_BᵀH_Bg_B/|g_B|²] — the expected directional
                # curvature along the batch gradient, DISTINCT from the
                # minibatch λmax above. (i) the step's own current minibatch
                # Rayleigh quotient (already computed by the ARPACK call
                # above; previously discarded); (ii) an m-minibatch average on
                # fresh training-distribution batches from the dedicated rng.
                spec_row["lhg_curmb"] = r["lambda_hat_grad"]
                vals = []
                for _ in range(_BS_SHARP_M):
                    idx_bs = rng_bs.integers(0, N, BS)
                    vals.append(_grad_rayleigh_on_batch(
                        model, Xtr[idx_bs], ytr[idx_bs], args.chunk_size))
                spec_row["lhg_bs_mean"] = float(np.mean(vals))
                spec_row["lhg_bs_std"] = float(np.std(vals))
            # Taylor residual of SAM's perturbed-point re-query (review
            # round 5): on the CURRENT training minibatch, at the arm's own
            # ρ — the nonlinear content separating full SAM from its
            # first-order (linSAM) expansion. Logged for the two arms that
            # sit on either side of that linearization.
            if args.opt in ("sam", "linsam", "sam_bnfreeze", "linsam_bnmatched",
                            "sam_interp", "linsam_neg"):
                spec_row.update(_taylor_residual_on_batch(
                    model, Xb, yb, args.rho, args.chunk_size))
            curv.pop("_evecs", None)
            curv.pop("_evals", None)
            xsam_alpha_g = xsam_corr_norm = xsam_cos_g_v = float("nan")
            rows.append(dict(step=step, train_loss=float(loss_val), eval_loss=tr_loss,
                              eval_acc=acc, fired_frac=fired_frac, **curv,
                              xsam_alpha_g=xsam_alpha_g, xsam_corr_norm=xsam_corr_norm,
                              xsam_cos_g_v=xsam_cos_g_v, **spec_row))
            elapse = time.time() - t0
            sps = (step + 1) / elapse
            eta_min = (STEPS - step - 1) / sps / 60 if sps else float("nan")
            extra = ""
            if plant_state is not None:
                md = plant_state.metrics()
                extra = (f" | λ_sense {md['lam_sensed']:.2f} div {md['divisor']:.2f}"
                         f" damp {md['damping']:+.3f} fire_frac {fired_frac:.3f}")
            elif rankkx_state is not None:
                md = rankkx_state.metrics()
                extra = (f" | λ̂₁ {md['lams_sensed'][0]:.2f} div₁ {md['divisors'][0]:.2f}"
                         f" damp₁ {md['dampings'][0]:+.3f} fire_frac {fired_frac:.3f}")
            print(
                f"step {step:4d} | loss {loss_val:.3f} acc {acc:.3f} | "
                f"λ_max(arpack) {curv['lambda_max']:.2f} (2/η={2/LR:g}) | "
                f"λ̂_g {curv['lambda_hat_grad']:+.2f} ritz {curv['lambda_max_arpack_ritz']:.3f}"
                f"{extra} | {sps:.1f} step/s eta {eta_min:.1f}min",
                flush=True)
        elif do_outcome_legacy:
            curv, warm_v = measure_curvature(model, base, Xeval, yeval, SHARP_NITER, warm_v=warm_v)
            with torch.no_grad():
                tr_loss = float(F.cross_entropy(model(Xeval), yeval))
                acc = eval_acc(model, Xte[:2000], yte[:2000])
            if rank_state:
                fired_frac = n_fired_cum / max(1, (step + 1) * args.k)
            elif xsam_state is not None:
                fired_frac = n_fired_cum / max(1, step + 1)
            else:
                fired_frac = float("nan")
            # X-SAM per-step diagnostics (NaN for non-xsam opts → uniform schema)
            if xsam_state is not None:
                xsam_alpha_g = float(xsam_state.alpha_g) if args.opt == "xsam_faithful" \
                    else float(getattr(xsam_state, "alpha", float("nan")))
                xsam_corr_norm = float(xsam_state.corr_norm) if args.opt == "xsam_faithful" \
                    else float("nan")
                xsam_cos_g_v = float(xsam_state.cos_g_v) if args.opt == "xsam_faithful" \
                    else float("nan")
            else:
                xsam_alpha_g = xsam_corr_norm = xsam_cos_g_v = float("nan")
            rows.append(dict(step=step, train_loss=float(loss_val), eval_loss=tr_loss,
                              eval_acc=acc, fired_frac=fired_frac, **curv,
                              xsam_alpha_g=xsam_alpha_g, xsam_corr_norm=xsam_corr_norm,
                              xsam_cos_g_v=xsam_cos_g_v))
            elapse = time.time() - t0
            sps = (step + 1) / elapse
            eta_min = (STEPS - step - 1) / sps / 60 if sps else float("nan")
            if rank_state:
                extra = f" | fired {rank_state.n_fired}/{args.k} (cum frac {fired_frac:.3f})"
            elif xsam_state is not None:
                extra = (f" | fired {n_fired_cum}/{step + 1} (cum frac {fired_frac:.3f})"
                         f" α_g {xsam_alpha_g:+.2f} |α_g| {abs(xsam_alpha_g):.2f}"
                         f" corr {xsam_corr_norm:.2f}")
            else:
                extra = ""
            print(
                f"step {step:4d} | loss {loss_val:.3f} acc {acc:.3f} | "
                f"λ_max {curv['lambda_max']:.2f} (2/η={2/LR:g}) | "
                f"λ̂_g {curv['lambda_hat_grad']:+.2f} cos_g {abs(curv['cos_grad_vtop']):.3f}"
                f"{extra} | {sps:.1f} step/s eta {eta_min:.1f}min",
                flush=True)

    if plant_jsonl is not None:
        plant_jsonl.close()

    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / "metrics.csv", index=False)
    # --- end-of-training full-test-set eval + final weights (200epf queue) ---
    # Runs ONCE after the loop, so per-step behavior above is untouched and
    # metrics.csv keeps its schema. final_acc stays the 2k-subset value for
    # backward compatibility; final_acc_full is the FULL test set (10k).
    acc_full = eval_acc(model, Xte, yte)
    manifest = dict(model=("resnet18-tv" if args.model == "resnet18" else "timm_vit_tiny"),
             n_params=n_params, opt=args.opt,
             bn_protocol=_BN_PROTOCOL.get(args.opt),
             dataset=args.dataset, n_classes=n_classes,
             lr=LR, momentum=MOM,
             bs=BS, steps=STEPS, seed=SEED,
             augment=bool(args.augment),
               rho=args.rho if args.opt in ("sam", "linsam", "xsam_faithful",
                                             "xsam_clamped", "sam_bnfreeze",
                                             "sgd_bn2x", "linsam_bnmatched",
                                             "sam_interp", "linsam_neg") else None,
              tau=args.tau if args.opt == "sam_interp" else None,
             alpha=args.alpha if args.opt == "xsam_faithful" else None,
             k=args.k if args.opt in ("rank1", "rankk", "rankkx") else None,
             plant_N=args.plant_N if args.opt in ("true_lmax", "amplify", "rankkx") else None,
             lam_targ=(args.lam_targ if args.lam_targ is not None else 2.0/LR) if args.opt in ("true_lmax", "rankkx") else None,
             edit=args.edit if args.opt == "rankkx" else None,
             amplify_strength=(args.amplify if args.opt == "rankkx" and args.edit == "amplify"
                               else (args.amplify if args.opt == "amplify" else None)),
             sense_current=bool(args.sense_current) if args.opt == "rankkx" else None,
             schedule=args.schedule,
             lr_min=args.lr_min if args.schedule == "cosine" else None,
             spectrum_k=args.spectrum_k if args.spectrum_k > 0 else None,
             multibatch=bool(args.multibatch) or None,
             eval_batch=args.eval_batch,
             outcome_arpack=bool(args.outcome_arpack),
             outcome_every=args.outcome_every if args.outcome_arpack else None,
              final_loss=float(df["train_loss"].iloc[-1]),
              final_acc=float(df["eval_acc"].iloc[-1]),
              final_acc_full=float(acc_full),
              lambda_max_final=float(df["lambda_max"].iloc[-1]),
              lambda_max_max=float(df["lambda_max"].max()),
              total_time_s=time.time() - t0)
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Persist final weights: state_dict includes BN running buffers.
    # r11: gated behind --save-final (previously unconditional). The save
    # happens after metrics.csv/manifest write and consumes no RNG, so
    # training dynamics and logged columns are unaffected either way.
    if args.save_final:
        torch.save({"model": model.state_dict(), "manifest": manifest},
                   RUN_DIR / "final_model.pt")
        print(f"final weights -> {RUN_DIR / 'final_model.pt'}")
    print(f"\n[done] -> {RUN_DIR}")
    print(f"λ_max: init={df['lambda_max'].iloc[0]:.2f} max={df['lambda_max'].max():.2f} "
          f"final={df['lambda_max'].iloc[-1]:.2f}  (2/η={2/LR})")
    print(f"final acc {df['eval_acc'].iloc[-1]:.3f} (2k) | full {acc_full:.3f} ({len(yte)})")
    print(f"total time {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
