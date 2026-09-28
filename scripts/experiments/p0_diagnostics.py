"""P0 diagnostics for the true-λ_max plant design (Reviewer #1 gate).

This script produces the three measurements the oracle + librarian need BEFORE
the main plant experiment can be parameterized, plus a built-in validation of
the new ARPACK tool:

  • ARPACK validation on a synthetic nn.Linear — top-3 eigenvalues vs the
    materialized full Hessian, relative-error check (< 1e-6 required).

  • Task B (P0b) — λ_max(B) SATURATION CURVE: at a warmed-up checkpoint, sweep
    the eval-batch size B ∈ {128, 256, 512, 1024, 2048, 5120, 10240, 51200} and
    find the smallest B whose λ_max is within 5 % of the B=51200 reference.
    That B_sat is the eval batch used for every subsequent λ_max measurement
    (so cross-trajectory comparisons are at a converged batch size, not a noisy
    small one).

  • Task C (P0c) — POWER-ITER vs ARPACK SCATTER: at 4 checkpoints
    (step ∈ {0, 500, 2000, last}) of a fresh deterministic vanilla-SGD run,
    measure λ_max with the EXISTING power-iteration estimator (the one the
    paper used so far) AND with ARPACK (ground truth). Report the systematic
    under-estimation of power iteration. This is the "why we had to fall back
    to λ̂_g as a proxy" evidence for the paper.

  • Task D (P0d) — VANILLA SGD TRUE λ_max TRAJECTORY: along the same fresh
    deterministic vanilla-SGD run, measure ARPACK λ_max (k=3, tol=1e-6) every
    `--sharp-every` steps on B_sat. Outputs the trajectory of (λ_1, λ_2, gap,
    λ̂_g). This is the baseline the plant's λ_max will be compared against to
    decide escape vs pin in the main experiment.

Determinism: full run uses torch.use_deterministic_algorithms(True) +
CUBLAS_WORKSPACE_CONFIG=:4096:8 (mirrors scripts/experiments/train_cifar100_resnet18.py),
so the trajectory is reproducible seed-to-seed and cross-method comparable.

Outputs (JSON):
    runs/p0/arpack_validation.json
    runs/p0/saturation_B.json                       (Task B table + B_sat)
    runs/p0/poweriter_vs_arpack.json                (Task C scatter)
    runs/p0/vanilla_sgd_lambda_max_trajectory.json  (Task D trajectory)

Usage (defaults match the existing runs/cifar100_r18_sgd_s0/ config):
    python scripts/experiments/p0_diagnostics.py --steps 5000 --sharp-every 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# MUST be set before the first CUDA call (deterministic matmul algo select).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.data.cifar import load_cifar
from steos.optim import SGD
from steos.sharpness import max_eigenvalue
from steos.sharpness.hessian import (
    _param_list,
    _flatten,
    _hvp_flat_chunked,
    _unflatten,
    hvp,
    max_eigenvalue_arpack,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = Path("runs/p0")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Per-chunk batch size for the chunked HVP. ResNet-18's conv double-backward
# (im2col buffer) is the memory bottleneck: at chunk=2048 the im2col alone is
# ~5 GB and tips the 16 GB GPU into OOM during ARPACK (verified). chunk=1024
# is the empirical sweet spot on a 16 GB card: full double-backward graph
# < ~6 GB, fits with headroom for the eval-batch tensors + ARPACK auxiliary
# vectors. Trade-off: smaller chunk ⇒ more HVPs per matvec ⇒ slower ARPACK.
DEFAULT_CHUNK = 1024


# ----------------------------------------------------------------------------
# Model (identical to scripts/experiments/train_cifar100_resnet18.py — same architecture so
# the λ_max values are directly comparable to the existing SGD run).
# ----------------------------------------------------------------------------
def build_resnet18(num_classes: int = 100) -> nn.Module:
    import torchvision.models as tvm

    net = tvm.resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


# ----------------------------------------------------------------------------
# BN freeze (paranoid copy of xsam_faithful.py L168-174): ARPACK does many
# forwards inside the matvec; in train mode those would pollute BN running
# stats. We set eval() upstream (which already prevents the updates) AND
# save/restore here as belt-and-braces so a caller mistake can't corrupt the
# trajectory.
# ----------------------------------------------------------------------------
def snapshot_bn(model):
    return [(m, m.running_mean.clone(), m.running_var.clone())
            for m in model.modules() if hasattr(m, "running_mean")]


def restore_bn(saved):
    for m, rm, rv in saved:
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)


# ----------------------------------------------------------------------------
# Single-HVP gradient Rayleigh quotient λ̂_g = gᵀHg / gᵀg — the existing
# reliable λ_max proxy. Used as a comparator alongside ARPACK λ_max.
# ----------------------------------------------------------------------------
def lambda_hat_grad(model, loss_fn, X, y, params, chunk_size=None):
    """λ̂_g = gᵀHg / gᵀg with chunked HVP for memory safety on large B."""
    out = model(X)
    loss = loss_fn(out, y)
    g = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
    g = [gi.detach() for gi in g]
    if chunk_size is None or chunk_size >= X.shape[0]:
        Hg = hvp(model, loss_fn, X, y, g, params)
    else:
        g_flat = _flatten(g)
        Hg_flat = _hvp_flat_chunked(
            model, loss_fn, X, y, g_flat, params, chunk_size=chunk_size)
        Hg = _unflatten(Hg_flat, params)
    gg = float(sum((gi * gi).sum().item() for gi in g))
    gHg = float(sum((gi * hi).sum().item() for gi, hi in zip(g, Hg)))
    return gHg / max(gg, 1e-30)


def make_eval_batch(Xtr, ytr, B, seed=123):
    """Deterministic fixed eval subset. Same seed ⇒ same indices ⇒ comparable
    λ_max across methods. For B values that exceed len(Xtr) (not here:
    CIFAR-100 train has 50000), we sample with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Xtr), B)
    return Xtr[idx], ytr[idx]


# ----------------------------------------------------------------------------
# Phase 0 — ARPACK validation on a synthetic problem.
# ----------------------------------------------------------------------------
def validate_arpack_synthetic() -> dict:
    """Compare ARPACK top-3 vs torch.linalg.eigh on a small nn.Linear(5,3)
    Hessian. Confirms (a) the ARPACK wrapper is correct, (b) the chunked HVP
    path is algebraically exact."""
    torch.manual_seed(42)
    np.random.seed(42)
    model = nn.Linear(5, 3).double()
    X = torch.randn(8, 5, dtype=torch.float64)
    y = torch.randn(8, 3, dtype=torch.float64)
    loss_fn = lambda p, t: F.mse_loss(p, t)
    params = _param_list(model)
    n = sum(p.numel() for p in params)

    # full Hessian via n basis-vector HVPs
    H = torch.zeros((n, n), dtype=torch.float64)
    col = 0
    for j, p in enumerate(params):
        for idx in np.ndindex(*p.shape):
            v = [torch.zeros_like(pp) for pp in params]
            e = torch.zeros_like(p)
            e[idx] = 1.0
            v[j] = e
            Hv = hvp(model, loss_fn, X, y, v, params)
            H[:, col] = torch.cat([h.detach().flatten() for h in Hv])
            col += 1
    H = 0.5 * (H + H.T)  # symmetrize (numerical guard)
    full_evals, _ = torch.linalg.eigh(H)  # ascending
    top3_full = full_evals[-3:].numpy()[::-1]  # descending

    evals_single, _ = max_eigenvalue_arpack(
        model, loss_fn, X, y, params, k=3, tol=1e-9)
    evals_chunk, _ = max_eigenvalue_arpack(
        model, loss_fn, X, y, params, k=3, tol=1e-9, chunk_size=3)

    rel_single = float(np.max(np.abs(evals_single - top3_full)
                              / np.maximum(np.abs(top3_full), 1e-12)))
    rel_chunk = float(np.max(np.abs(evals_chunk - top3_full)
                             / np.maximum(np.abs(top3_full), 1e-12)))
    print(f"[validate] full   top-3: {top3_full}")
    print(f"[validate] arpack top-3: {evals_single}  rel={rel_single:.2e}")
    print(f"[validate] chunk  top-3: {evals_chunk}   rel={rel_chunk:.2e}")
    assert rel_single < 1e-6, f"ARPACK validation FAILED: rel err {rel_single}"
    assert rel_chunk < 1e-6, f"chunked HVP validation FAILED: rel err {rel_chunk}"
    print("[validate] OK — ARPACK matches full Hessian to < 1e-6 rel err.")
    return {
        "n_params": int(n),
        "full_top3": top3_full.tolist(),
        "arpack_top3": evals_single.tolist(),
        "arpack_chunked_top3": evals_chunk.tolist(),
        "max_rel_err_single": rel_single,
        "max_rel_err_chunked": rel_chunk,
        "passed": rel_single < 1e-6 and rel_chunk < 1e-6,
    }


# ----------------------------------------------------------------------------
# Task B — λ_max(B) saturation curve.
# ----------------------------------------------------------------------------
# Full requested set per task spec. The two largest (10240, 51200) are
# expensive (chunk=1024 ⇒ 10 / 50 chunks per matvec); the curve stops early
# once saturation is detected, so they are typically skipped in practice.
B_SET_FULL = [128, 256, 512, 1024, 2048, 5120, 10240, 51200]
SAT_TOL = 0.05  # 5 % within-reference threshold
SAT_RUN = 3     # how many consecutive B must agree within SAT_TOL to stop early


def warmup_sgd(model, Xtr, ytr, n_steps, lr, mom, wd, bs, seed):
    """Run n_steps of plain SGD to populate BN running stats + move θ off the
    random init. This is the "warmup checkpoint" the task specifies for Task B
    (the goal is a representative checkpoint, not the very initial random θ
    where BN stats are uninformative)."""
    base = SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    rng = np.random.default_rng(seed)
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(n_steps):
        idx = rng.integers(0, len(Xtr), bs)
        Xb, yb = Xtr[idx], ytr[idx]
        base.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xb), yb)
        loss.backward()
        base.step()
    return float(loss.item())


def task_B_saturation(model, Xtr, ytr, max_B=5120, b_tol=1e-4, b_k=2) -> dict:
    """Sweep B ∈ {128...max_B} (subset of the full task-spec set, capped for
    runtime) and find the smallest B whose λ_max saturates. Uses tol=1e-4 + k=2
    (vs D's tol=1e-6 + k=3): the B-curve only needs the saturation *shape*,
    not high-precision eigenvalues, and large B values would be intractable
    at full ARPACK precision (B=51200 ⇒ 50 chunked-HVPs × ~30 matvecs ≈ 25 min
    at tol=1e-6; ~5 min at tol=1e-4)."""
    B_set = [B for B in B_SET_FULL if B <= max_B]
    print(f"\n{'='*70}\n=== Task B: λ_max(B) saturation curve  "
          f"B ∈ {B_set}  (tol={b_tol}, k={b_k})\n{'='*70}")
    ce = lambda p, t: F.cross_entropy(p, t)
    params = _param_list(model)
    model.eval()
    saved_bn = snapshot_bn(model)
    table = []
    try:
        for B in B_set:
            Xe, ye = make_eval_batch(Xtr, ytr, B, seed=123)
            chunk = min(DEFAULT_CHUNK, B)
            t0 = time.time()
            try:
                evals, _ = max_eigenvalue_arpack(
                    model, ce, Xe, ye, params, k=b_k, tol=b_tol,
                    chunk_size=chunk, maxiter=200)
                lam1, lam2 = (float(evals[0]), float(evals[1])
                              if len(evals) > 1 else float("nan"))
            except Exception as exc:
                lam1, lam2 = float("nan"), float("nan")
                print(f"  [warn] B={B} ARPACK failed: {exc!r}")
            try:
                lhg = lambda_hat_grad(model, ce, Xe, ye, params, chunk_size=chunk)
            except Exception as exc:
                lhg = float("nan")
                print(f"  [warn] B={B} λ̂_g failed: {exc!r}")
            dt = time.time() - t0
            row = {"B": B, "lambda_max": lam1, "lambda_2": lam2,
                   "lambda_hat_g": float(lhg), "chunk_size": chunk,
                   "time_s": dt}
            table.append(row)
            print(f"  B={B:6d} | λ_max={lam1:9.3f}  λ_2={lam2:9.3f}  "
                  f"λ̂_g={lhg:9.3f}  chunk={chunk}  ({dt:.1f}s)", flush=True)
            restore_bn(saved_bn)
            torch.cuda.empty_cache()  # defragment between B values

            # early-stop: SAT_RUN consecutive B values within SAT_TOL ⇒ done
            if len(table) >= SAT_RUN:
                recent = table[-SAT_RUN:]
                lams = [r["lambda_max"] for r in recent
                        if r["lambda_max"] == r["lambda_max"]]
                if len(lams) == SAT_RUN:
                    hi, lo = max(lams), min(lams)
                    if (hi - lo) / max(abs(hi), 1e-12) <= SAT_TOL:
                        print(f"  [early-stop] last {SAT_RUN} B values agree "
                              f"within {SAT_TOL*100:.0f}% ({lo:.2f}–{hi:.2f}); "
                              f"saturation reached.")
                        break
    finally:
        model.train()

    # B_sat: smallest B whose λ_max is within SAT_TOL of the LARGEST measured.
    ref = table[-1]["lambda_max"]
    B_sat = None
    for row in table:
        if (row["lambda_max"] == row["lambda_max"] and
            abs(row["lambda_max"] - ref) / max(abs(ref), 1e-12) <= SAT_TOL):
            B_sat = row["B"]
            break
    if B_sat is None:
        B_sat = max_B
        print(f"  [warn] no B saturated within {SAT_TOL*100:.0f}%; "
              f"defaulting B_sat=max_B={max_B}")
    print(f"\n  >> B_sat = {B_sat}  (first B within {SAT_TOL*100:.0f}% of "
          f"B={table[-1]['B']} λ_max={ref:.3f})")
    return {"table": table, "B_ref_value": ref, "B_ref_size": int(table[-1]["B"]),
            "B_sat": int(B_sat), "sat_tol": SAT_TOL, "b_tol": b_tol, "b_k": b_k}


# ----------------------------------------------------------------------------
# Tasks C + D — combined deterministic vanilla-SGD run.
# ----------------------------------------------------------------------------
def run_deterministic_sgd(
    steps: int,
    sharp_every: int,
    B_sat: int,
    C_steps: set,
    Xtr, ytr, Xte, yte,
    seed: int = 0,
    lr: float = 0.1,
    mom: float = 0.9,
    wd: float = 5e-4,
    bs: int = 128,
    arpack_tol: float = 1e-6,
    arpack_k: int = 3,
):
    """One fresh deterministic vanilla-SGD run. At every sharp step we record
    ARPACK λ_max(k=3, tol) + λ̂_g on B_sat → the trajectory (Task D). At C-target
    steps we additionally record the power-iteration λ_max → the scatter (Task C).

    Returns (traj_rows, c_scatter_rows)."""
    print(f"\n{'='*70}\n=== Tasks C+D: deterministic vanilla-SGD run\n"
          f"    steps={steps}  sharp_every={sharp_every}  B_sat={B_sat}  "
          f"C_steps={sorted(C_steps)}\n{'='*70}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_resnet18(num_classes=100).to(DEV)
    base = SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    ce = lambda p, t: F.cross_entropy(p, t)
    params = _param_list(model)

    Xe_d, ye_d = make_eval_batch(Xtr, ytr, B_sat, seed=123)
    chunk = min(DEFAULT_CHUNK, B_sat)

    rng = np.random.default_rng(seed)
    traj = []
    c_scatter = []
    t0 = time.time()

    for step in range(steps + 1):
        is_sharp = (step % sharp_every == 0) or (step in C_steps) or (step == steps)
        if is_sharp:
            # ---- measure ARPACK + λ̂_g on B_sat (eval mode, frozen BN) ----
            model.eval()
            saved_bn = snapshot_bn(model)
            try:
                evals, _ = max_eigenvalue_arpack(
                    model, ce, Xe_d, ye_d, params,
                    k=arpack_k, tol=arpack_tol, chunk_size=chunk)
                lam1, lam2 = float(evals[0]), float(evals[1])
                gap_pct = (lam1 - lam2) / max(abs(lam1), 1e-12) * 100.0
                lhg = lambda_hat_grad(model, ce, Xe_d, ye_d, params, chunk_size=chunk)
                with torch.no_grad():
                    tr_loss = float(F.cross_entropy(model(Xe_d), ye_d))
                    acc = float((model(Xte[:1000]).argmax(1) == yte[:1000]).float().mean())
                row = {"step": step, "lambda_max": lam1, "lambda_2": lam2,
                       "gap_pct": gap_pct, "lambda_hat_g": lhg,
                       "train_loss_eval_batch": tr_loss, "eval_acc_1k": acc}
                traj.append(row)
                print(f"  step {step:5d} | λ_max={lam1:8.3f}  λ_2={lam2:8.3f}  "
                      f"gap={gap_pct:5.2f}% | λ̂_g={lhg:8.3f}  "
                      f"| tr={tr_loss:.3f} acc={acc:.3f} "
                      f"| {1.0/max(time.time()-t0,1e-9):.2f} meas/s", flush=True)

                # ---- Task C: also measure power-iter at C target steps ----
                if step in C_steps or step == steps:
                    lam_pi, _, n_hvp_pi = max_eigenvalue(
                        model, ce, Xe_d, ye_d, n_iter=200, tol=1e-4,
                        return_vec=True, return_iters=True)
                    under_pct = (lam1 - float(lam_pi)) / max(abs(lam1), 1e-12) * 100.0
                    c_scatter.append({
                        "step": step,
                        "lambda_max_power": float(lam_pi),
                        "lambda_max_power_n_hvp": int(n_hvp_pi),
                        "lambda_max_arpack": lam1,
                        "lambda_2_arpack": lam2,
                        "lambda_hat_g": lhg,
                        "underestimate_pct": float(under_pct),
                    })
                    print(f"           C scatter | power={float(lam_pi):8.3f} "
                          f"(n_hvp={n_hvp_pi})  underest={under_pct:+.2f}%", flush=True)
            finally:
                restore_bn(saved_bn)
                model.train()

        if step >= steps:
            break

        # ---- plain SGD step ----
        idx = rng.integers(0, len(Xtr), bs)
        Xb, yb = Xtr[idx], ytr[idx]
        base.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xb), yb)
        loss.backward()
        base.step()

    dt = time.time() - t0
    print(f"\n  [done] full D run: {dt/60:.1f} min  ({len(traj)} sharp measurements)")
    return traj, c_scatter


# ----------------------------------------------------------------------------
# Briefing — written to runs/p0/BRIEFING.md and printed to stdout.
# ----------------------------------------------------------------------------
def write_briefing(B_result, traj, c_scatter, val_result, args):
    import statistics as st

    table = B_result["table"]
    B_sat = B_result["B_sat"]
    ref_B = B_result["B_ref_value"]
    ref_Bsz = B_result["B_ref_size"]

    # power-iter systematic under-estimation (mean across C scatter)
    unders = [r["underestimate_pct"] for r in c_scatter
              if r["underestimate_pct"] == r["underestimate_pct"]]  # filter NaN
    mean_under = float(st.mean(unders)) if unders else float("nan")

    # vanilla SGD true λ_max trajectory stats (skip NaNs)
    lams = [r["lambda_max"] for r in traj
            if r["lambda_max"] == r["lambda_max"]]
    lhgs = [r["lambda_hat_g"] for r in traj
            if r["lambda_hat_g"] == r["lambda_hat_g"]]
    gaps = [r["gap_pct"] for r in traj if r["gap_pct"] == r["gap_pct"]]

    lam_mean = float(st.mean(lams)) if lams else float("nan")
    lam_med = float(st.median(lams)) if lams else float("nan")
    lam_min = float(min(lams)) if lams else float("nan")
    lam_max = float(max(lams)) if lams else float("nan")
    lhg_mean = float(st.mean(lhgs)) if lhgs else float("nan")
    gap_med = float(st.median(gaps)) if gaps else float("nan")

    # exclude step 0 from "steady" stats (init transient; λ_max huge at random init)
    steady = [r for r in traj if r["step"] > 0
              and r["lambda_max"] == r["lambda_max"]]
    steady_lams = [r["lambda_max"] for r in steady]
    steady_mean = float(st.mean(steady_lams)) if steady_lams else float("nan")
    steady_med = float(st.median(steady_lams)) if steady_lams else float("nan")
    steady_min = float(min(steady_lams)) if steady_lams else float("nan")
    steady_max = float(max(steady_lams)) if steady_lams else float("nan")

    lines = []
    lines.append("# P0 Diagnostics — Briefing")
    lines.append("")
    lines.append(f"## ARPACK validation (synthetic nn.Linear(5,3))")
    lines.append(f"- full-vs-ARPACK rel err (single): "
                 f"{val_result['max_rel_err_single']:.2e}")
    lines.append(f"- full-vs-ARPACK rel err (chunked): "
                 f"{val_result['max_rel_err_chunked']:.2e}")
    lines.append(f"- PASS: {val_result['passed']}  (threshold 1e-6)")
    lines.append("")
    lines.append(f"## Task B — λ_max(B) saturation")
    lines.append("")
    lines.append("| B | λ_max | λ_2 | λ̂_g | chunk |")
    lines.append("|---|-------|-----|------|-------|")
    for r in table:
        lines.append(f"| {r['B']} | {r['lambda_max']:.3f} | "
                     f"{r['lambda_2']:.3f} | {r['lambda_hat_g']:.3f} | "
                     f"{r['chunk_size']} |")
    lines.append("")
    lines.append(f"**B_sat = {B_sat}**  (first B within 5% of "
                 f"B={ref_Bsz} λ_max={ref_B:.3f})")
    lines.append("")
    lines.append(f"## Task C — power-iter vs ARPACK scatter")
    lines.append("")
    lines.append("| step | λ_max(power) | n_hvp | λ_max(ARPACK) | underest % |")
    lines.append("|------|--------------|-------|----------------|------------|")
    for r in c_scatter:
        lines.append(f"| {r['step']} | {r['lambda_max_power']:.3f} | "
                     f"{r['lambda_max_power_n_hvp']} | "
                     f"{r['lambda_max_arpack']:.3f} | "
                     f"{r['underestimate_pct']:+.2f}% |")
    lines.append("")
    lines.append(f"**Power-iter mean systematic under-estimation: "
                 f"{mean_under:+.2f}%**  "
                 f"(positive ⇒ power iter UNDER-estimates true λ_max)")
    lines.append("")
    lines.append(f"## Task D — vanilla SGD true λ_max trajectory")
    lines.append("")
    lines.append(f"- measurement batch B_sat = {B_sat}, "
                 f"ARPACK tol={args.arpack_tol}, k={args.arpack_k}")
    lines.append(f"- measurements: {len(traj)} (every {args.sharp_every} steps)")
    lines.append(f"- λ_max ALL: mean={lam_mean:.3f} median={lam_med:.3f} "
                 f"min={lam_min:.3f} max={lam_max:.3f}")
    lines.append(f"- λ_max STEADY (step>0): mean={steady_mean:.3f} "
                 f"median={steady_med:.3f} min={steady_min:.3f} "
                 f"max={steady_max:.3f}")
    lines.append(f"- λ̂_g mean: {lhg_mean:.3f}")
    lines.append(f"- (λ_1-λ_2)/λ_1 gap median: {gap_med:.2f}%  "
                 f"(small gap ⇒ power iter needs many iters to separate ⇒ "
                 f"explanation for the systematic under-estimation above)")
    lines.append("")
    lines.append(f"## Verdict for the plant experiment")
    lines.append(f"- **eval batch for all sensing**: B = {B_sat}")
    lines.append(f"- **plant λ_max target**: match vanilla-SGD steady median "
                 f"≈ {steady_med:.1f}; compare to 2/η = {2/args.lr:g}")
    lines.append(f"- **escape/pin interpretation**: the plant's λ_max must "
                 f"exceed the vanilla trajectory's typical λ_max "
                 f"(steady median {steady_med:.1f}, max {steady_max:.1f}) "
                 f"for the damping to actually engage (2/η = {2/args.lr:g}).")

    text = "\n".join(lines)
    (RUN_DIR / "BRIEFING.md").write_text(text)
    print("\n" + "=" * 70 + "\nBRIEFING\n" + "=" * 70 + "\n" + text + "\n")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000,
                    help="vanilla-SGD run length for Task D")
    ap.add_argument("--sharp-every", type=int, default=200,
                    help="measure ARPACK λ_max every N steps (Task D)")
    ap.add_argument("--c-steps", type=int, nargs="+", default=[0, 500, 2000, 5000],
                    help="checkpoints for Task C scatter")
    ap.add_argument("--warmup-steps", type=int, default=50,
                    help="Task B warmup SGD steps before the B-curve")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--arpack-tol", type=float, default=1e-6)
    ap.add_argument("--arpack-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-b", action="store_true",
                    help="skip Task B (use --b-sat instead)")
    ap.add_argument("--b-sat", type=int, default=None,
                    help="override B_sat (with --skip-b)")
    ap.add_argument("--max-B", type=int, default=5120,
                    help="cap on B for the saturation sweep (large B is "
                         "expensive: chunked HVPs scale linearly with B)")
    ap.add_argument("--skip-d", action="store_true",
                    help="skip Task D (still runs validate + B + C-warmup-only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. ARPACK validation on synthetic (always)
    val_result = validate_arpack_synthetic()
    (RUN_DIR / "arpack_validation.json").write_text(json.dumps(val_result, indent=2))

    # 2. Load CIFAR-100
    print(f"\nloading CIFAR-100 on {DEV} ...", flush=True)
    Xtr, ytr, _, Xte, yte = load_cifar(n_classes=100, device=DEV)
    print(f"  train {len(Xtr)}  test {len(Xte)}", flush=True)

    # 3. Task B (warmup model + sweep B)
    if args.skip_b:
        assert args.b_sat is not None, "--skip-b requires --b-sat"
        B_result = {"B_sat": int(args.b_sat), "B_ref_value": float("nan"),
                    "B_ref_size": int(args.b_sat), "table": [],
                    "sat_tol": SAT_TOL, "b_tol": float("nan"), "b_k": 0}
        print(f"\n[skip-B] using B_sat = {args.b_sat}")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model_B = build_resnet18(num_classes=100).to(DEV)
        print(f"\nTask B warmup ({args.warmup_steps} SGD steps) ...", flush=True)
        warmup_loss = warmup_sgd(
            model_B, Xtr, ytr, n_steps=args.warmup_steps,
            lr=args.lr, mom=args.mom, wd=args.wd, bs=args.bs, seed=args.seed)
        print(f"  warmup final batch loss = {warmup_loss:.3f}", flush=True)
        B_result = task_B_saturation(model_B, Xtr, ytr, max_B=args.max_B)
        (RUN_DIR / "saturation_B.json").write_text(json.dumps(B_result, indent=2))
        del model_B
        torch.cuda.empty_cache()

    B_sat = int(B_result["B_sat"])

    # 4. Tasks C+D (fresh deterministic vanilla-SGD run)
    if args.skip_d:
        print("\n[skip-D] Tasks C+D skipped; only ARPACK validation + B done.")
        return
    C_steps = set(args.c_steps) | {args.steps}
    traj, c_scatter = run_deterministic_sgd(
        steps=args.steps, sharp_every=args.sharp_every, B_sat=B_sat,
        C_steps=C_steps, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, seed=args.seed,
        lr=args.lr, mom=args.mom, wd=args.wd, bs=args.bs,
        arpack_tol=args.arpack_tol, arpack_k=args.arpack_k)
    (RUN_DIR / "vanilla_sgd_lambda_max_trajectory.json").write_text(
        json.dumps({"B_sat": B_sat, "config": vars(args),
                    "trajectory": traj}, indent=2))
    (RUN_DIR / "poweriter_vs_arpack.json").write_text(
        json.dumps({"B_sat": B_sat, "scatter": c_scatter}, indent=2))

    # 5. Briefing
    write_briefing(B_result, traj, c_scatter, val_result, args)


if __name__ == "__main__":
    main()
