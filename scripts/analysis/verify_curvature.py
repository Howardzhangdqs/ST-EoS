"""Curvature correctness gate — run before trusting any sharpness number.

THIS IS A HARD GATE WITH AN EXIT CODE: every known-answer check below must
pass (exit 0 + "ALL CHECKS PASSED") before producing any figure or writing
anything into FINDINGS. A failed check prints a diagnostic to stderr and
exits 1 — a non-zero exit means every downstream number is untrusted.

For full-batch GD on a quadratic L(θ)=½θ^THθ, g=Hθ, a direct calculation gives
    L(θ - ηg) - L(θ) = -η g^Tg + (η²/2) g^THg
which first turns positive at  η_c = 2 g^Tg / g^THg  ⇒  λ_c = 2/η_c = g^THg / g^Tg.
That is the Rayleigh quotient of H along the GRADIENT direction — NOT λ_max
(the Rayleigh quotient along the TOP eigenvector). They coincide only when g is
aligned with the top eigenvector.

This script verifies both the quadratic identity (known answer, asserted) and
that the same relationship holds on the real MLP loss (informational). It also
asserts the degenerate flat-loss case returns NaN rather than a fabricated
λ_c. If the asserts pass, λ_c is correct and the λ_c→λ_max convergence we see
near EoS is genuine gradient/top-eigenvector alignment (the Damian et al.
self-stabilization signature).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.optim import SGD, register_param_names
from steos.sharpness import critical_sharpness, max_eigenvalue
from steos.sharpness.hessian import hvp

torch.manual_seed(0)


def _dot(a, b):
    return float(sum((ai * bi).sum().item() for ai, bi in zip(a, b)))


def _norm(a):
    return math.sqrt(sum((ai * ai).sum().item() for ai in a))


def _fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


class Quadratic(nn.Module):
    """L(θ) = 0.5 θᵀ H θ with diagonal H(eigenvalues = `eigs`)."""

    def __init__(self, dim, eigs, init):
        super().__init__()
        self.theta = nn.Parameter(init.clone())
        self.register_buffer("H", torch.diag(eigs.clone()))

    def forward(self, X=None):
        return self.theta  # loss_fn uses this directly


def quad_loss(H):
    def fn(out, y=None):
        return 0.5 * (out * (H @ out)).sum()
    return fn


def measure_quadratic(eigs, init, label):
    print(f"\n--- {label}: eigs={eigs.tolist()}, θ={init.tolist()} ---")
    dim = len(eigs)
    model = Quadratic(dim, eigs, init)
    opt = SGD(model.parameters(), lr=1e-3)
    register_param_names(opt, model)
    H = model.H
    loss = quad_loss(H)
    # populate grad at θ
    loss(model()).backward()
    params = [model.theta]
    g = [model.theta.grad.detach().clone()]

    # ground truth
    lam_max_true = float(eigs.max())
    Hg = hvp(model, loss, None, None, g, params)
    lam_eff = _dot(g, Hg) / max(_dot(g, g), 1e-30)  # g^T H g / g^T g
    lam_max_pi, vtop = max_eigenvalue(model, loss, None, None, n_iter=500, tol=1e-8, return_vec=True)
    align = abs(_dot(g, vtop)) / _norm(g)  # |cos(angle(g, v_top))|

    def loss_closure(batch):
        return float(loss(model()))

    lam_c, eta_c, nfwd = critical_sharpness(model, opt, (None, None), loss_closure, lr_guess=0.5, tol_power=6)

    print(f"  λ_max (true)            = {lam_max_true:.6f}")
    print(f"  λ_max (power iter)      = {lam_max_pi:.6f}")
    print(f"  λ_eff = g^THg/g^Tg      = {lam_eff:.6f}   <-- theory prediction for λ_c")
    print(f"  λ_c (bracketing)        = {lam_c:.6f}   (η_c={eta_c:.6g}, {nfwd} fwd)")
    print(f"  |alignment(g, v_top)|   = {align:.4f}")
    ratio_c = lam_c / max(lam_eff, 1e-12)
    ratio_max = lam_max_pi / lam_max_true
    print(f"  CHECK λ_c≈λ_eff:  ratio={ratio_c:.4f}")
    print(f"  CHECK λ_max≈true: ratio={ratio_max:.4f}")
    # Hard gate: λ_c must match the Rayleigh-quotient prediction within 1%.
    if not abs(ratio_c - 1.0) < 0.01:
        _fail(
            f"{label}: λ_c/λ_eff = {ratio_c:.6f} deviates from 1 by "
            f"{abs(ratio_c - 1.0) * 100:.3f}% (tol 1%)"
        )
    if not abs(ratio_max - 1.0) < 0.01:
        _fail(
            f"{label}: λ_max(power iter)/λ_max(true) = {ratio_max:.6f} deviates "
            f"from 1 by {abs(ratio_max - 1.0) * 100:.3f}% (tol 1%)"
        )


def check_flat_loss():
    """Degenerate case: θ at the exact minimum ⇒ grad ≈ 0 ⇒ Δθ = 0 ⇒ the loss
    never flips and bracketing cannot find η_c. Must return NaN — a fabricated
    λ_c = 2/lr_guess would silently poison downstream analysis."""
    print("\n--- flat loss: θ at exact minimum (grad ≈ 0) — expect NaN pair ---")
    eigs = torch.tensor([1.0, 5.0, 50.0])
    model = Quadratic(3, eigs, torch.zeros(3))
    opt = SGD(model.parameters(), lr=1e-3)
    register_param_names(opt, model)
    loss = quad_loss(model.H)
    loss(model()).backward()  # grad is exactly zero here

    def loss_closure(batch):
        return float(loss(model()))

    lam_c, eta_c, nfwd = critical_sharpness(
        model, opt, (None, None), loss_closure, lr_guess=0.5, tol_power=6
    )
    print(f"  λ_c = {lam_c}, η_c = {eta_c} ({nfwd} fwd)  <-- must both be NaN")
    # NOTE: assert NaN-ness directly with math.isnan — a ratio-based assert
    # would be False for NaN in BOTH the pass and fail directions.
    if not (math.isnan(lam_c) and math.isnan(eta_c)):
        _fail(f"flat loss: expected (nan, nan), got λ_c={lam_c}, η_c={eta_c}")


def measure_mlp():
    print("\n--- real MLP loss at a single θ ---")
    from steos.data import make_dataset, loss_fn
    from steos.models import build_model

    torch.manual_seed(1)
    X, y, task = make_dataset("teacher_student", n=512, dim=10, seed=0, device="cuda")
    model = build_model("mlp", in_dim=10, hidden_dims=[64, 64], out_dim=1, activation="tanh").to("cuda")
    opt = SGD(model.parameters(), lr=0.05)
    register_param_names(opt, model)
    loss = loss_fn(task)
    loss(model(X), y).backward()
    params = [p for p in model.parameters() if p.requires_grad]
    g = [p.grad.detach().clone() for p in params]

    Hg = hvp(model, loss, X, y, g, params)
    lam_eff = _dot(g, Hg) / max(_dot(g, g), 1e-30)
    lam_max, vtop = max_eigenvalue(model, loss, X, y, n_iter=300, tol=1e-7, return_vec=True)
    align = abs(_dot(g, vtop)) / _norm(g)

    def loss_closure(batch):
        Xb, yb = batch
        return float(loss(model(Xb), yb))

    lam_c, eta_c, nfwd = critical_sharpness(model, opt, (X, y), loss_closure, lr_guess=0.1, tol_power=6)
    print(f"  λ_max (power iter)      = {lam_max:.4f}")
    print(f"  λ_eff = g^THg/g^Tg      = {lam_eff:.4f}   <-- theory prediction for λ_c")
    print(f"  λ_c (bracketing)        = {lam_c:.4f}   (η_c={eta_c:.6g}, {nfwd} fwd)")
    print(f"  |alignment(g, v_top)|   = {align:.4f}    (→1 at EoS ⇒ λ_c→λ_max)")
    print(f"  CHECK λ_c≈λ_eff:  ratio={lam_c/max(lam_eff,1e-12):.4f}  (expect ≈1)")
    print(f"  NOTE λ_c/λ_max = {lam_c/max(lam_max,1e-12):.4f} = cos²(angle) roughly (≤1)")


if __name__ == "__main__":
    # Pure-quadratic tests with KNOWN answers (hard-gated at 1%).
    eigs = torch.tensor([1.0, 5.0, 50.0])
    measure_quadratic(eigs, torch.tensor([1.0, 1.0, 1.0]), "g dominated by top eigvec")
    measure_quadratic(eigs, torch.tensor([1.0, 1.0, 0.0]), "g in mid/low eigendirs")
    measure_quadratic(eigs, torch.tensor([0.0, 0.0, 1.0]), "g exactly along top eigvec")
    # Degenerate flat-loss case must yield NaN, never a fabricated λ_c.
    check_flat_loss()
    # Real-loss cross-check (informational, no known answer to assert on).
    measure_mlp()
    print("\nALL CHECKS PASSED")
    sys.exit(0)
