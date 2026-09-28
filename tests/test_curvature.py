"""CPU-only correctness tests for the λ_c (critical sharpness) estimator.

For plain GD on a quadratic L(θ)=½θᵀHθ, the critical LR is exactly
η_c = 2 gᵀg / gᵀHg, i.e. λ_c = 2/η_c equals the Rayleigh quotient of H along
the gradient. These tests check that identity (1% tolerance) plus the two
degenerate cases that must propagate NaN instead of a pseudo-value:
a flat loss (zero gradient, bracketing can never flip) and a diverged run
(loss_init already NaN, short-circuit at entry). Runs in seconds on CPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steos.optim import SGD, register_param_names
from steos.sharpness import critical_sharpness


class Quadratic(nn.Module):
    """L(θ) = 0.5 θᵀ H θ with diagonal H (eigenvalues = `eigs`)."""

    def __init__(self, dim, eigs, init):
        super().__init__()
        self.theta = nn.Parameter(init.clone())
        self.register_buffer("H", torch.diag(eigs.clone()))

    def forward(self, X=None):
        return self.theta


def _dot(a, b):
    return float(sum((ai * bi).sum().item() for ai, bi in zip(a, b)))


def _measure(eigs, init):
    """Run critical_sharpness on the quadratic and return (λ_c, η_c, λ_eff)."""
    model = Quadratic(len(eigs), eigs, init)
    opt = SGD(model.parameters(), lr=1e-3)
    register_param_names(opt, model)

    def loss(out, y=None):
        return 0.5 * (out * (model.H @ out)).sum()

    loss(model()).backward()  # populate grad at θ
    g = model.theta.grad.detach().clone()
    gdot = _dot([g], [g])
    # Rayleigh quotient gᵀHg/gᵀg is undefined for g = 0 (flat-loss case).
    lam_eff = _dot([g], [model.H @ g]) / gdot if gdot > 0 else float("nan")

    def loss_closure(batch):
        return float(loss(model()))

    lam_c, eta_c, nfwd = critical_sharpness(
        model, opt, (None, None), loss_closure, lr_guess=0.5, tol_power=6
    )
    return lam_c, eta_c, nfwd, lam_eff


@pytest.mark.parametrize(
    "eigs,init",
    [
        (torch.tensor([1.0, 5.0, 50.0]), torch.tensor([1.0, 1.0, 1.0])),  # g mixes all dirs
        (torch.tensor([1.0, 5.0, 50.0]), torch.tensor([0.0, 0.0, 1.0])),  # g along top eigvec → λ_c=λ_max
        (torch.tensor([2.0, 3.0, 8.0, 20.0]), torch.tensor([1.0, 2.0, 0.5, 0.0])),  # low/mid dirs
    ],
)
def test_critical_sharpness_matches_rayleigh(eigs, init):
    lam_c, eta_c, nfwd, lam_eff = _measure(eigs, init)
    assert math.isfinite(lam_c), f"λ_c not finite: {lam_c}"
    assert abs(lam_c / lam_eff - 1.0) < 0.01, (
        f"λ_c={lam_c:.6f} vs Rayleigh prediction λ_eff={lam_eff:.6f} "
        f"(deviation {abs(lam_c / lam_eff - 1.0) * 100:.3f}%, tol 1%)"
    )


def test_flat_loss_returns_nan():
    # θ at the exact minimum ⇒ grad = 0 ⇒ Δθ = 0 ⇒ loss never flips ⇒ the
    # bracketing loop must exhaust and report NaN (NOT 2/lr_guess).
    lam_c, eta_c, nfwd, _ = _measure(
        torch.tensor([1.0, 5.0, 50.0]), torch.zeros(3)
    )
    # Use math.isnan, never a ratio: NaN comparisons are False in both
    # directions, so a ratio-based assert would be vacuous here.
    assert math.isnan(lam_c), f"flat loss: expected λ_c=NaN, got {lam_c}"
    assert math.isnan(eta_c), f"flat loss: expected η_c=NaN, got {eta_c}"


def test_diverged_loss_returns_nan():
    # loss_init already NaN (diverged run) → short-circuit at entry.
    model = Quadratic(3, torch.tensor([1.0, 5.0, 50.0]), torch.tensor([1.0, 1.0, 1.0]))
    opt = SGD(model.parameters(), lr=1e-3)
    register_param_names(opt, model)

    def loss(out, y=None):
        return 0.5 * (out * (model.H @ out)).sum()

    loss(model()).backward()

    def loss_closure(batch):
        return float("nan")  # simulates an already-diverged trajectory

    lam_c, eta_c, nfwd = critical_sharpness(
        model, opt, (None, None), loss_closure, lr_guess=0.5, tol_power=6
    )
    assert math.isnan(lam_c), f"diverged loss: expected λ_c=NaN, got {lam_c}"
    assert math.isnan(eta_c), f"diverged loss: expected η_c=NaN, got {eta_c}"
    assert nfwd == 0, f"diverged loss: expected 0 forward passes, got {nfwd}"
