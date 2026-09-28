"""EoS-Precond-rank-k — rank-k curvature preconditioning along the top-k
Hessian eigenvectors. Generalizes ``eos_precond.py`` from rank-1 to rank-k.

Hypothesis under test (Claim 1 generalized): damping the top-k sharp directions
does NOT let the optimizer escape the EoS threshold 2/η for k ≪ n, because
sharpness migrates to the undamped complement and λ_max re-pins at 2/η.

Mechanism (mirrors the rank-1 self-gating design, generalized over k orthonormal
directions V = [v_1, …, v_k]):
    λ̂_i      = v_iᵀ H v_i             (Rayleigh quotient, top-k estimate)
    divisor_i = max(λ̂_i / λ_targ, 1),  λ_targ = 2/η
    g_eff     = g − Σ_i (gᵀ v_i) v_i (1 − 1/divisor_i)

Because V is orthonormal (maintained by modified Gram-Schmidt), the parallel
components decouple and the rank-1 in-place gradient edit generalizes cleanly:
subtract each over-sharp projection, scaled by its own self-gating divisor. The
max(·, 1) clamp is self-gating: a direction below 2/η gets divisor 1 (≡ plain
SGD along that axis), so the method only fires once a direction actually crosses
the EoS boundary — early misalignment is harmless (same argument as rank-1).

CLAMP-FREE mode (``clamp=False``, bridge experiment): on ResNet-18 the C3
plant's self-gating clamp is dormant — it senses mini-batch sharpness ≈ 2/η, so
``divisor = max(λ̂/(2/η), 1) ≈ 1`` and it rarely fires (rank-1 ≈ SGD). A
reviewer asks whether this inertness is a CLAMP ARTIFACT (the clamp hides a real
effect) or a REAL batch-sharpness phenomenon. ``clamp=False`` removes the
``max(·, 1)`` gate entirely: ``divisor = λ̂/λ_targ`` (raw, may be < 1), every
direction fires every step, and the factor ``f = 1 − 1/divisor`` is clipped to
the half-open interval [−1, 1) for numerical safety (when λ̂ ≪ λ_targ the raw
factor → −∞). This is NOT "always damp" semantics: when the divisor is < 1
(λ̂ < λ_targ) the factor is negative, so the edit flips the sign of the
α v_i term — a bounded sign-flip amplification rather than damping (exact
magnitudes in the step-function docstring). If rank-1 then stays ≈ SGD the
inertness is a real batch-sharpness phenomenon; if it sharp-escapes like the
MLP accurate-v_top the clamp was hiding it. The ``clamp=True`` (default) path
is UNCHANGED.

Top-k estimation: warm-started block power iteration with modified Gram-Schmidt
(MGS) orthonormalization, ``n_sweeps`` sweeps per step (default 3 ⇒ ~3k HVPs /
step). V persists in the state across steps for a strong warm start (the EoS
subspace drifts slowly along the trajectory). Each sweep, for i in 1..k:
Hv = H v_i; orthogonalize Hv against v_1..v_{i-1} (MGS); normalize → new v_i;
λ̂_i is the Rayleigh quotient v_iᵀ H v_i of the pre-sweep v_i (which by the last
sweep is the converged eigenvector estimate).

Performance: the loss-gradient graph is built ONCE per step (``create_graph``)
and reused for all ``n_sweeps·k`` second-backwards (Pearlmutter Hv), and the
block iteration runs on FLAT 1-D parameter vectors with GPU-resident scalars —
each MGS projection is a single ``torch.dot`` + ``add_`` (not a per-parameter
loop), and there is exactly ONE host sync per step (the diagnostic batch). This
makes a k=10 step ~2× cheaper than the naïve per-call ``hvp`` / per-tensor-``.item``
formulation, which matters for the 5-seed sweep.

Implemented (like ``eos_precond_step``) as a SAM-style step FUNCTION wrapping a
base SGD optimizer: it edits ``p.grad`` in place, then calls
``base_optimizer.step()``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from ..sharpness.hessian import _param_list

__all__ = ["EoSPrecondRankKState", "eos_precond_rankk_step"]


# --------------------------------------------------------------------------- #
# flat-parameter geometry helpers                                              #
# (a "vector" is a 1-D torch.Tensor of length N = Σ p.numel(); slicing into    #
#  per-parameter views is free, so MGS / Rayleigh quotients / normalization    #
#  each cost a single kernel instead of a per-parameter Python loop.)          #
# --------------------------------------------------------------------------- #
def _build_index(params: List[torch.Tensor]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (numel-per-param, (start,end)-slice-pairs) for the flat layout."""
    starts: List[int] = []
    ends: List[int] = []
    s = 0
    for p in params:
        starts.append(s)
        s += p.numel()
        ends.append(s)
    return [p.numel() for p in params], list(zip(starts, ends))


def _init_random_orthonormal_flat(n: int, k: int, ref: torch.Tensor) -> List[torch.Tensor]:
    """k random orthonormal flat vectors (one MGS pass)."""
    V: List[torch.Tensor] = []
    for _ in range(k):
        v = torch.randn(n, device=ref.device, dtype=ref.dtype)
        for b in V:
            v = v - torch.dot(v, b) * b
        v = v / v.norm().clamp(min=1e-12)
        V.append(v)
    return V


# --------------------------------------------------------------------------- #
# state                                                                        #
# --------------------------------------------------------------------------- #
class EoSPrecondRankKState:
    """Carries the top-k orthonormal eigenvector estimates V = [v_1, …, v_k]
    (each a FLAT 1-D tensor of length N = Σ p.numel(), jointly orthonormal)
    across steps and exposes per-direction diagnostics from the most recent step.

    Attributes:
        k: rank.
        V: list of k orthonormal flat vectors (None until the first step).
        step_count: number of steps taken.
        lam_hats: length-k list of Rayleigh-quotient sharpness estimates λ̂_i.
        cos_g_v: length-k list of |gᵀ v_i| / ‖g‖ alignments with the gradient.
        divisors: length-k list of per-direction damping divisors.
        n_fired: how many of the k divisors fired (> 1) on the last step.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k: int = k
        self.V: Optional[List[torch.Tensor]] = None
        self.step_count: int = 0
        # diagnostics from the most recent step
        self.lam_hats: List[float] = [0.0] * k
        self.cos_g_v: List[float] = [0.0] * k
        self.divisors: List[float] = [1.0] * k
        self.n_fired: int = 0

    def metrics(self) -> dict:
        return dict(
            lam_hats=list(self.lam_hats),
            cos_g_v=list(self.cos_g_v),
            divisors=list(self.divisors),
            n_fired=self.n_fired,
            lam_hat_top=self.lam_hats[0] if self.k else 0.0,
        )


# --------------------------------------------------------------------------- #
# step                                                                         #
# --------------------------------------------------------------------------- #
def eos_precond_rankk_step(
    model: torch.nn.Module,
    base_optimizer: torch.optim.Optimizer,
    loss_fn,
    X,
    y,
    state: EoSPrecondRankKState,
    lr: float,
    k: Optional[int] = None,
    lambda_targ: Optional[float] = None,
    n_sweeps: int = 3,
    clamp: bool = True,
    amplify: Optional[float] = None,
    fixed_divisor: Optional[float] = None,
) -> float:
    """One rank-k EoS-Precond step. See module docstring.

    Args:
        model, base_optimizer, loss_fn, X, y: the usual training objects.
        state: an ``EoSPrecondRankKState`` holding V from the previous step
            (warm start); the first step initializes V with k random orthonormal
            flat tensors.
        lr: base learning rate η (used only to set λ_targ = 2/η if not given).
        k: rank (defaults to ``state.k``).
        lambda_targ: per-direction sharpness ceiling (default 2/η).
        n_sweeps: block-power-iteration sweeps per step (3 ⇒ ~3k HVPs/step).
        clamp: if True (DEFAULT), apply the self-gating ``max(λ̂/λ_targ, 1)``
            clamp — a direction at/below λ_targ is left untouched (behavior
            UNCHANGED from the original implementation). If False, run
            CLAMP-FREE: no ``max(·, 1)`` gate, every direction fires every
            step, and the divisor is the raw ratio ``λ̂/λ_targ`` (may be < 1).
            The factor ``f = 1 − 1/divisor`` is clipped to ``[-1, 1)`` for
            numerical safety (when ``λ̂ ≪ λ_targ`` the raw factor → −∞). This
            is NOT "always damp": when the divisor < 1 (λ̂ < λ_targ) the
            factor is NEGATIVE and the edit flips the sign of the α v_i
            component — a bounded sign-flip AMPLIFICATION, not damping. The
            subtracted edit has magnitude ``|α·f| ≤ |α|`` (max at f = −1),
            and the resulting v-component after the edit is ``α·(1−f)`` with
            magnitude ``≤ 2|α|`` (max at f = −1, where the component
            doubles). The clip is what prevents unbounded anti-damping
            (divergence); contrast the un-clipped lagged plant, which
            diverges (MLP, step ~540). This is the bridge experiment testing
            whether the ResNet-18 C3 plant's inertness is a clamp artifact
            or a real batch-sharpness phenomenon.
        amplify: if set (e.g. ``0.5``), AMPLIFICATION mode — the sign-flip
            control of Proposition ``prop:rank1-no-flat-escape``: the edit is
            the FIXED curvature-amplifying ``g_eff = g + amplify·(gᵀv̂)v̂``
            along each estimated eigendirection, applied unconditionally
            every step (γ = −amplify < 0 in the paper's g′ = g − γ(gᵀv̂)v̂
            notation). The 2-D reduction predicts the period-2 equilibrium
            descends to λ* = 2/((1+amplify)η) < 2/η (flat escape), with the
            same x-amplitude — the mirror image of damping's y* > 0.
            Overrides ``clamp``; sensing/warm-start machinery is identical to
            the damping arm so the two are directly comparable.
        fixed_divisor: if set (e.g. ``1.5``), FIXED-DIVISOR mode — the causal
            control that de-confounds the preconditioned-sharpness invariant:
            the self-gating ``d = λ̂/λ_targ`` FEEDBACK is removed and the
            damping divisor is held at the constant ``d_fix`` along each
            sensed direction, applied unconditionally every step (edit
            ``g_eff = g − α v̂ (1 − 1/d_fix)``; γ = 1 − 1/d_fix constant).
            With γ constant the 2-D period-2 fixed point
            y* = 2γ/((1−γ)η) is NO LONGER vacuous (with the adaptive rule
            substituting γ reduces it to the identity y* = y): it makes the
            non-trivial point prediction S* = 2 d_fix/η for the RAW
            equilibrium, which the experiment tests across a d_fix grid.
            Overrides ``clamp``; incompatible with ``amplify``.

    Returns:
        The loss value (float) at θ_t (before the step).
    """
    if k is None:
        k = state.k
    if lambda_targ is None:
        lambda_targ = 2.0 / lr
    if fixed_divisor is not None:
        if fixed_divisor <= 0.0:
            raise ValueError(f"fixed_divisor must be > 0, got {fixed_divisor}")
        if amplify is not None:
            raise ValueError("fixed_divisor and amplify are mutually exclusive")

    params = _param_list(model)
    _, slices = _build_index(params)
    n_flat = sum(p.numel() for p in params)
    base_optimizer.zero_grad(set_to_none=True)

    # --- build the loss-gradient graph ONCE and reuse for all HVPs ----------
    # (the naïve per-call hvp rebuilds fwd + create_graph backward every time;
    #  for block power iteration at k=10, 3 sweeps ⇒ 30 HVPs/step, so reusing
    #  the graph is much faster. p.grad is set from g for the base SGD step.)
    out = model(X)
    loss = loss_fn(out, y)
    loss_val = float(loss.item())
    g = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    with torch.no_grad():
        for p, gi in zip(params, g):
            p.grad = gi.detach()
    g_flat = torch.cat([gi.reshape(-1) for gi in g])     # 1-D, carries graph

    def hv_flat(v_flat: torch.Tensor) -> torch.Tensor:
        # Pearlmutter Hv via the retained graph: ONE dot (g·v) then second-back.
        gdv = torch.dot(g_flat, v_flat)                  # scalar w/ graph
        Hv = torch.autograd.grad(gdv, params, retain_graph=True)  # tuple
        return torch.cat([h.reshape(-1) for h in Hv])    # 1-D

    # --- initialize / warm-start the subspace estimate V (flat, orthonormal) -
    if state.V is None or len(state.V) != k or state.V[0].numel() != n_flat:
        V = _init_random_orthonormal_flat(n_flat, k, g_flat)
    else:
        V = state.V

    # --- warm-started block power iteration with MGS (n_sweeps sweeps) -------
    # GPU-resident: every projection / Rayleigh quotient / normalization is a
    # single torch.dot or elementwise op on 1-D tensors (no per-parameter loop,
    # no host sync). λ̂_i is the Rayleigh quotient of the pre-sweep v_i (which by
    # the final sweep equals the converged eigenvector estimate).
    zero = g_flat.new_zeros(())
    lams_t: List[torch.Tensor] = [zero.clone() for _ in range(k)]
    for _ in range(n_sweeps):
        V_new: List[torch.Tensor] = []
        for i in range(k):
            Hv = hv_flat(V[i])                            # 1-D (H v_i)
            lams_t[i] = torch.dot(V[i], Hv)               # 0-dim Rayleigh quotient
            for b in V_new:                               # MGS vs v_1..v_{i-1}
                Hv = Hv - torch.dot(Hv, b) * b            # no host sync
            V_new.append(Hv / Hv.norm().clamp(min=1e-12))  # guard degeneracy
        V = V_new
    state.V = V

    # --- self-gating damping along each of the k orthonormal directions -------
    grad_flat = torch.cat([p.grad.reshape(-1) for p in params])  # type: ignore[union-attr]
    alpha_t = [torch.dot(grad_flat, V[i]) for i in range(k)]
    gnorm2_t = torch.dot(grad_flat, grad_flat)
    # ONE batched host sync for [‖g‖², α_0..α_{k-1}, λ̂_0..λ̂_{k-1}]
    batch = torch.stack([gnorm2_t] + alpha_t + lams_t)
    vals = batch.tolist()
    gnorm2 = vals[0]
    alphas = vals[1:1 + k]
    lams = [max(x, 0.0) for x in vals[1 + k:1 + 2 * k]]
    gnorm = math.sqrt(max(gnorm2, 0.0))
    if fixed_divisor is not None:
        # FIXED-DIVISOR mode (causal control): no λ̂ feedback, the divisor is a
        # constant d_fix applied unconditionally. Tests the 2-D point
        # prediction S* = 2·d_fix/η for the RAW equilibrium without the
        # controller identity that the adaptive rule carries.
        divisors = [float(fixed_divisor)] * k
        fired = [True] * k
    elif clamp:
        # self-gating: a direction at/below λ_targ gets divisor 1 (≡ plain SGD
        # along that axis), so the plant only fires once a direction actually
        # crosses the EoS boundary. (Default; behavior UNCHANGED.)
        divisors = [max(lams[i] / lambda_targ, 1.0) for i in range(k)]
        fired = [divisors[i] > 1.0 + 1e-9 for i in range(k)]
    else:
        # CLAMP-FREE mode (bridge experiment): NO max(·,1) gate, ALWAYS fire.
        # Tests whether the ResNet-18 C3 plant inertness is a CLAMP ARTIFACT or
        # a real batch-sharpness phenomenon. divisors carry the raw λ̂/λ_targ
        # ratio (may be < 1); the correction is applied every step.
        divisors = [lams[i] / lambda_targ for i in range(k)]
        fired = [True] * k

    # diagnostics are computed on the ORIGINAL gradient (before the in-place edit)
    state.lam_hats = lams
    state.cos_g_v = [abs(alphas[i]) / max(gnorm, 1e-30) for i in range(k)]
    state.divisors = list(divisors)
    state.n_fired = sum(1 for f in fired if f)

    # in-place gradient edit: g_eff = g − Σ_i α_i v_i (1 − 1/divisor_i)
    # (V orthonormal ⇒ the parallel components decouple; subtract each one with
    #  its own self-gating divisor — exactly the rank-1 formula summed over k.
    #  V[i] is flat; slice it back into per-parameter views to edit p.grad.)
    with torch.no_grad():
        for i in range(k):
            if amplify is not None:
                # AMPLIFICATION mode (sign-flip control): the FIXED edit
                # g_eff = g + amplify·α_i v_i (γ = −amplify < 0), unconditional.
                c = -amplify * alphas[i]
            elif not fired[i]:
                continue
            elif fixed_divisor is not None or clamp:
                # self-gating damping (default) and the fixed-divisor causal
                # control share the same edit formula c = α(1 − 1/d); only the
                # provenance of d differs (λ̂/λ_targ with a max(·,1) floor vs.
                # the constant d_fix).
                c = alphas[i] * (1.0 - 1.0 / divisors[i])
            else:
                # CLAMP-FREE factor: f = 1 − 1/d with the RAW divisor d =
                # λ̂/λ_targ (may be < 1), clipped to the half-open interval
                # [-1, 1) for numerical safety (when λ̂ ≪ λ_targ, d → 0 and
                # the raw factor → −∞). This is NOT "always damp": when
                # d < 1 (λ̂ < λ_targ) the factor is negative and the edit
                # c = α·f FLIPS THE SIGN of the α v_i term — a bounded
                # sign-flip AMPLIFICATION, not damping. Precise magnitudes:
                # the subtracted edit |c| = |α·f| ≤ |α| (max at f = −1), and
                # the resulting v-component after the edit is α·(1−f) with
                # magnitude ≤ 2|α| (max at f = −1). The clip is what
                # prevents unbounded anti-damping (divergence); contrast the
                # un-clipped lagged plant which diverges (MLP, step ~540).
                # (Guard d==0 to avoid float div-by-zero; d ≥ 0 since λ̂ ≥ 0.)
                d = divisors[i]
                factor_i = -1.0 if d <= 0.0 else 1.0 - 1.0 / d
                if factor_i < -1.0:
                    factor_i = -1.0
                elif factor_i >= 1.0:
                    factor_i = 1.0 - 1e-12
                c = alphas[i] * factor_i
            vi = V[i]
            for p, (a, b) in zip(params, slices):
                p.grad.sub_(c * vi[a:b].view(p.shape))  # type: ignore[union-attr]

    base_optimizer.step()  # plain SGD reads the (possibly edited) p.grad
    state.step_count += 1
    return loss_val
