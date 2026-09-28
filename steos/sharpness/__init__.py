"""Curvature measurement: classic λ_max (Hessian power iteration) and scalable
λ_c (critical sharpness, Kalra et al.). `make_sharpness_hook` produces the
callable the Trainer invokes at θ_t (before the step) to record curvature."""

from .critical_lr import atParams, critical_sharpness, measure_critical_lr, virtual_step
from .hessian import _param_list, hvp, max_eigenvalue, top_k_eigenvalues
from .rayleigh import rayleigh_proxies

import math

import numpy as np


def make_sharpness_hook(
    optimizer,
    loss_fn,
    kinds=("lambda_max", "lambda_c"),
    n_iter: int = 200,
    tol: float = 1e-4,
    lr_guess: float = 1e-3,
    tol_power: int = 4,
    warm_start_vec: bool = True,
):
    """Build a hook: `hook(step, model, X, y, loss_value) -> dict`.

    Measured at the CURRENT θ (before the optimizer step), with param.grad
    already populated by the just-run backward — so Δθ is the update about to
    be applied, matching the λ_c definition. The top eigenvector and the η_c
    guess are warm-started across calls (they drift slowly along the trajectory).

    kinds: "lambda_max" (power iter), "lambda_c" (critical sharpness line search),
           "rayleigh" (single-HVP Δθ/g Rayleigh proxies — requires "lambda_max").

    Note: the power-iteration lambda_max estimator used here has a measured
    warm-start bias — see the ARPACK cross-validation (finding F32) before
    trusting sub-percent accuracies.
    """
    kinds = set(kinds)
    if "rayleigh" in kinds and "lambda_max" not in kinds:
        # rayleigh_proxies needs v_top, which only lambda_max produces
        kinds.add("lambda_max")
    state = {"v": None, "lr_guess": float(lr_guess)}

    def hook(step, model, X, y, loss_value):
        rec: dict = {"loss_at_measure": float(loss_value)}

        if "lambda_max" in kinds:
            v0 = state["v"] if warm_start_vec else None
            ret = max_eigenvalue(
                model, loss_fn, X, y, n_iter=n_iter, tol=tol, return_vec=True,
                return_iters=True, v_init=v0,
            )
            lam, v, n_hvp = ret  # type: ignore[misc]
            if warm_start_vec:
                state["v"] = v
            rec["lambda_max"] = float(lam)
            rec["lambda_max_n_hvp"] = int(n_hvp)
            # persist flattened top eigenvector for Phase-3 (Δθ vs v_top) analysis
            rec["vtop_flat"] = np.concatenate(
                [vi.detach().cpu().numpy().ravel() for vi in v]
            ).astype(np.float32)

            if "rayleigh" in kinds:
                params = _param_list(model)
                rec.update(
                    {k: float(val) for k, val in
                     rayleigh_proxies(model, optimizer, loss_fn, X, y, v, params).items()}
                )

        if "lambda_c" in kinds:
            def loss_closure(batch):
                Xb, yb = batch
                return float(loss_fn(model(Xb), yb))

            lam_c, eta_c, nfwd = critical_sharpness(
                model, optimizer, (X, y), loss_closure,
                lr_guess=state["lr_guess"], tol_power=tol_power,
            )
            rec["lambda_c"] = float(lam_c)
            rec["eta_c"] = float(eta_c)
            rec["lambda_c_n_fwd"] = int(nfwd)
            if not math.isnan(eta_c):
                # NaN would poison the warm start for every later measurement.
                state["lr_guess"] = float(eta_c)  # warm-start next measurement

        return rec

    return hook


__all__ = [
    "max_eigenvalue",
    "top_k_eigenvalues",
    "hvp",
    "rayleigh_proxies",
    "critical_sharpness",
    "measure_critical_lr",
    "atParams",
    "virtual_step",
    "make_sharpness_hook",
]
