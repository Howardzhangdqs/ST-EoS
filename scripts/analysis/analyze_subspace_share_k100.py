"""Cumulative minibatch-gradient energy share s_k at a saved final checkpoint
+ Monte-Carlo m-convergence check for the batch sharpness estimator
(review ask: top-20 share < 0.3% invites "why stop at 20 — CIFAR-100 needs
~class-count directions"; this measures s_k for k ∈ {20, 50, 100} directly).

Pipeline (offline, one checkpoint; precedent: scripts/verify_lambda_max_
arpack.py — model/data rebuilt from the run's manifest):

  1. load <run>/final_model.pt ({"model": state_dict, "manifest": ...}),
     rebuild build_resnet18(num_classes=manifest["n_classes"]) and
     load_cifar — the SAME loaders the trainer used;
  2. rebuild the deterministic fixed eval batch the trainer's sensing used
     (np.random.default_rng(123).integers(0, N, eval_batch) —
     train_cifar100_resnet18.py lines 546–551, _ARP_EVAL_SEED = 123);
  3. ARPACK top-k eigenpairs of the eval-mode fixed-batch Hessian
     (steos.sharpness.hessian.max_eigenvalue_arpack — the same sensing infra
     run_cifar100_det_arpack_threeway.sh / the rankkx plants use), eval mode
     with BN running stats snapshotted/restored (snapshot_bn/restore_bn from
     steos.optim.true_lmax_plant, per the max_eigenvalue_arpack caller
     contract). COST WARNING: tol=1e-4 at k=100 needs a large Krylov basis
     (ncv ~ k+32 vectors × 11.2M params × 8 B ≈ 12 GB RAM) and several
     hundred HVPs — budget ~1–3 GPU-h per checkpoint;
  4. over m fresh training minibatches (bs 128, eval-mode current-batch
     definition — mirrors the in-run lhg convention
     _grad_rayleigh_on_batch: eval mode, BN frozen, chunk-wise gradient
     accumulation, then ONE Pearlmutter HVP):
       s_k = E_B[ Σ_{i≤k} (g_B·v̂_i)² / ‖g_B‖² ]   for each k in --ks,
       lhg(B) = g_BᵀH_B g_B / ‖g_B‖²               (batch sharpness),
     and report lhg mean±SE at m ∈ --ms (nested: the first m of the
     max(ms) draws) as the Monte-Carlo estimator-convergence check.

Conventions vs in-run quantities (disclosed): v̂_i come from the FIXED eval
batch (B=512, seed 123) like the plant sensing; g_B and H_B are EVAL-mode on
fresh un-augmented training minibatches like the lhg columns. The in-run
canonical plant share (~2e-4 at top-20) was measured on the TRAIN-mode
training gradient along the plant's warm-started sensed directions — the
checkpoint numbers may legitimately differ.

Usage:
    .venv/bin/python scripts/analysis/analyze_subspace_share_k100.py \
        --run runs/cifar100_r18_sgd_200epf_s0 --ks 20,50,100 --ms 4,16,64 \
        --out figures/subspace_share_k100.csv
    # smoke (k=20 only, ~10–15 min):
    .venv/bin/python scripts/analysis/analyze_subspace_share_k100.py \
        --run runs/cifar100_r18_sgd_200epf_s0 --ks 20 --ms 4,16 \
        --out figures/subspace_share_k100_smoke_sgd200epf_s0.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from steos.data.cifar import load_cifar                     # noqa: E402
from steos.optim.true_lmax_plant import restore_bn, snapshot_bn  # noqa: E402
from steos.sharpness.hessian import (                       # noqa: E402
    _hvp_flat_chunked,
    _param_list,
    max_eigenvalue_arpack,
)
from train_cifar100_resnet18 import build_resnet18           # noqa: E402


def batch_grad_evalmode(model, X, y, params, chunk):
    """g_B in EVAL mode, chunk-wise accumulation with create_graph=False —
    mirrors _grad_rayleigh_on_batch (train_cifar100_resnet18.py lines
    297–334). Caller must already be in eval mode with BN snapshotted."""
    ce_ = lambda p, yy: F.cross_entropy(p, yy)              # noqa: E731
    n = X.shape[0]
    cs = min(chunk or n, n)
    g_flat = None
    for i in range(0, n, cs):
        Xc, yc = X[i:i + cs], y[i:i + cs]
        nc = Xc.shape[0]
        loss = ce_(model(Xc), yc) * (nc / n)
        gs = torch.autograd.grad(loss, params)
        gf = torch.cat([gi.reshape(-1) for gi in gs])
        g_flat = gf if g_flat is None else g_flat + gf
    return g_flat.detach()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory (final_model.pt)")
    ap.add_argument("--ks", default="20,50,100",
                    help="comma-separated cumulative-share cutoffs (default 20,50,100)")
    ap.add_argument("--ms", default="4,16,64",
                    help="comma-separated minibatch counts for the lhg "
                         "m-convergence check (default 4,16,64; nested — "
                         "max(ms) batches are drawn and reused)")
    ap.add_argument("--out", default="figures/subspace_share_k100.csv")
    ap.add_argument("--tol", type=float, default=1e-4, help="ARPACK tol")
    ap.add_argument("--ncv", type=int, default=None,
                    help="ARPACK ncv (None: scipy default for k<=40, else "
                         "k+32 to cap the Krylov basis RAM)")
    ap.add_argument("--bs", type=int, default=128, help="minibatch size")
    ap.add_argument("--batch-seed", type=int, default=20260921,
                    help="rng seed for the fresh minibatch draws (dedicated "
                         "offline stream; NOT the in-run rng_bs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="chunked-HVP batch cap (256 = shared-GPU safe)")
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    ms = sorted({int(x) for x in args.ms.split(",") if x.strip()})
    m_max = max(ms)
    k_solve = max(ks)

    run_dir = Path(args.run)
    ckpt = torch.load(run_dir / "final_model.pt", map_location=args.device,
                      weights_only=False)
    man = ckpt["manifest"]
    if man.get("model") != "resnet18-tv":
        raise SystemExit(f"unexpected model {man.get('model')!r} — this script "
                         "rebuilds resnet18-tv only")

    print(f"[load] {run_dir.name}: opt={man['opt']} steps={man['steps']} "
          f"seed={man['seed']} schedule={man['schedule']} "
          f"final_acc_full={man.get('final_acc_full')}", flush=True)
    model = build_resnet18(num_classes=man["n_classes"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    params = _param_list(model)
    n_flat = sum(p.numel() for p in params)
    print(f"[model] {n_flat/1e6:.2f}M params restored from final_model.pt",
          flush=True)

    Xtr, ytr, _, Xte, yte = load_cifar(n_classes=man["n_classes"],
                                        device=args.device)
    N = len(Xtr)
    # deterministic eval batch — trainer's sensing/outcome convention
    rng_eval = np.random.default_rng(123)
    idx_eval = rng_eval.integers(0, N, int(man["eval_batch"]))
    Xeval, yeval = Xtr[idx_eval], ytr[idx_eval]
    ce = lambda p, y: F.cross_entropy(p, y)                 # noqa: E731

    # ---- ARPACK top-k on the fixed eval batch (eval mode, BN frozen) -------
    ncv = args.ncv if args.ncv is not None else (
        None if k_solve <= 40 else k_solve + 32)
    saved_bn = snapshot_bn(model)
    t0 = time.time()
    try:
        evals, evecs = max_eigenvalue_arpack(
            model, ce, Xeval, yeval, params, k=k_solve, tol=args.tol,
            chunk_size=args.chunk_size, ncv=ncv)
    finally:
        restore_bn(saved_bn)
    t_arpack = time.time() - t0
    evals = np.asarray(evals, float)
    V = np.ascontiguousarray(evecs[:, :k_solve].astype(np.float32))  # (n, k)
    # Ritz residual of the top pair (quality gate, trainer convention)
    v1 = torch.from_numpy(V[:, 0]).to(args.device)
    Hv1 = _hvp_flat_chunked(model, ce, Xeval, yeval, v1, params, args.chunk_size)
    ritz = float((Hv1 - float(evals[0]) * v1).norm().item()
                 / max(abs(float(evals[0])), 1e-30))
    print(f"[arpack] k={k_solve} tol={args.tol} ncv={ncv}  "
          f"lam1={evals[0]:.3f} lam2={evals[1]:.3f} "
          f"lam{k_solve}={evals[-1]:.3f}  ritz_top={ritz:.2e}  "
          f"({t_arpack/60:.1f} min)", flush=True)

    # ---- fresh minibatches: s_k + lhg ---------------------------------------
    rng = np.random.default_rng(args.batch_seed)
    V_gpu = torch.from_numpy(V).to(args.device)      # (n, k) float32
    shares = np.zeros((m_max, len(ks)), float)       # Σ_{i≤k}(g·v̂_i)²/‖g‖²
    lhgs = np.zeros(m_max, float)
    t1 = time.time()
    for j in range(m_max):
        idx = rng.integers(0, N, args.bs)
        Xb, yb = Xtr[idx], ytr[idx]
        saved_bn = snapshot_bn(model)
        try:
            g = batch_grad_evalmode(model, Xb, yb, params, args.chunk_size)
            gg = float(g.pow(2).sum().item())
            proj2 = (V_gpu.T @ g).pow(2).cpu().numpy()     # (g·v̂_i)²
            cum = np.cumsum(proj2) / max(gg, 1e-30)
            shares[j] = cum[[k - 1 for k in ks]]
            Hg = _hvp_flat_chunked(model, ce, Xb, yb, g, params, args.chunk_size)
            lhgs[j] = float((g * Hg).sum().item()) / max(gg, 1e-30)
        finally:
            restore_bn(saved_bn)
        print(f"  batch {j+1}/{m_max}: ‖g‖={float(np.sqrt(gg)):.3f} "
              f"lhg={lhgs[j]:.3f} "
              + " ".join(f"s_{k}={shares[j, i]:.3e}" for i, k in enumerate(ks)),
              flush=True)
    t_mb = time.time() - t1

    # ---- outputs -------------------------------------------------------------
    rows = [dict(kind="arpack", k=k_solve, m=float("nan"), n_batches=1,
                 mean=float(evals[0]), se=float("nan"),
                 note=f"lam1={evals[0]:.4f} lam2={evals[1]:.4f} "
                      f"lam_k={evals[-1]:.4f} ritz_top={ritz:.2e} "
                      f"tol={args.tol} ncv={ncv} t_min={t_arpack/60:.1f}")]
    for i, k in enumerate(ks):
        v = shares[:, i]
        rows.append(dict(kind="share_s_k", k=k, m=m_max, n_batches=m_max,
                         mean=float(v.mean()),
                         se=float(v.std(ddof=1) / np.sqrt(len(v)))
                         if len(v) > 1 else 0.0,
                         note="cumulative minibatch-grad energy share "
                              "E_B[sum_(i<=k)(g_B.v_i)^2/||g_B||^2]"))
    for m in ms:
        v = lhgs[:m]
        rows.append(dict(kind="lhg_m", k=float("nan"), m=m, n_batches=m,
                         mean=float(v.mean()),
                         se=float(v.std(ddof=1) / np.sqrt(len(v)))
                         if len(v) > 1 else 0.0,
                         note="batch sharpness gHg/gg at m minibatches "
                              "(nested first-m of the draws)"))
    df = pd.DataFrame(rows)
    df.insert(0, "run", run_dir.name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"=> {out}")
    print(f"[time] arpack {t_arpack/60:.1f} min + {m_max} minibatches "
          f"{t_mb/60:.1f} min; per-batch cost {t_mb/m_max:.1f} s "
          f"(share+m-lhg probes scale linearly in m)")


if __name__ == "__main__":
    main()
