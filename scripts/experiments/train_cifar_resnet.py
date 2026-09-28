"""Core generalization test: does EoS (and the Gap-A findings) reproduce on a
NORMALIZED convnet (PreAct-ResNet + BatchNorm) on real images (CIFAR-10)?

Mini-batch SGD with momentum, lr=0.1 (⇒ 2/η=20). Curvature λ_max is measured
periodically in EVAL mode on a fixed batch (so BN uses running stats → a
deterministic Hessian). This is the Candidate-1 HEADLINE regime: under momentum
Δθ ≠ −ηg, so the update-direction Rayleigh quotient λ̂_Δθ can diverge from the
gradient-direction one λ̂_g (≡ λ_c) — the novel case.

Records per sharpness step: λ_max (warm-started power iter), the single-HVP
Rayleigh proxies λ̂_grad & λ̂_update with their |cos(·, v_top)| alignments, and
per-group parameter norms (the BN-γ sharpness proxy baseline, Finding 14).

--warm-start-track (default OFF; without it the run is byte-identical to
before) additionally records the reviewer-requested one-HVP warm-started
power-iteration baseline: persistent v_ws, and per measurement point on the
SAME sharp batch: w = H v_ws (single HVP); λ̂_ws = v_ws·w/‖v_ws‖²;
v_ws ← w/‖w‖. New CSV columns: lambda_hat_ws, cos_ws_vtop (the vector USED
for the estimate vs the measured v_top), t_hvp_g_s / t_hvp_ws_s
(time.perf_counter around each estimator's HVP). v_ws is initialized at the
first measurement from a dedicated torch.Generator seeded with --seed → the
training trajectory is RNG-identical to a flag-off run.

Usage:
    HTTPS_PROXY=http://127.0.0.1:7890 python scripts/experiments/train_cifar_resnet.py \
        --run-name cifar_rayleigh --lr 0.1 --steps 2500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.data.cifar import load_cifar10
from steos.models import build_model
from steos.optim import SGD, register_param_names
from steos.sharpness import max_eigenvalue, rayleigh_proxies
from steos.sharpness.hessian import _param_list, hvp
from steos.monitor.snapshots import SnapshotManager

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def eval_acc(model, Xv, yv, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(Xv), bs):
        pred = model(Xv[i:i + bs]).argmax(1)
        correct += int((pred == yv[i:i + bs]).sum())
    model.train()
    return correct / len(Xv)


@torch.no_grad()
def group_norms(model):
    """Combined ‖·‖ of parameter groups — candidates for the BN-regime
    sharpness proxy (Finding 13 follow-up: which survives BatchNorm?)."""
    g: dict[str, list[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if name.endswith(".weight") and ("bn" in name or "norm" in name):
            key = "bn_gamma"
        elif name.endswith(".weight") and "conv" in name:
            key = "conv_w"
        elif name == "fc.weight":
            key = "fc_w"
        elif name.endswith(".bias") and ("bn" in name):
            key = "bn_beta"
        elif name.endswith(".weight"):
            key = "other_w"
        else:
            key = "other"
        g.setdefault(key, []).append(p.detach().reshape(-1))
    return {k: float(torch.cat(v).norm().item()) for k, v in g.items()}


def _warm_start_step(model, ce, Xeval, yeval, params, v_used):
    """ONE-HVP warm-started power-iteration update (reviewer comparator).

    w = H v_ws; λ̂_ws = v_ws·w / ‖v_ws‖²; v_ws ← w/‖w‖. `v_used` is the
    state carried from the PREVIOUS measurement. Returns
    (lambda_hat_ws, cos_vs_vtop, v_new, hvp_seconds). cos is computed by the
    caller (needs v_top); here we only do the HVP + algebra.
    """
    t0 = time.perf_counter()
    w = hvp(model, ce, Xeval, yeval, v_used, params)
    t_hvp = time.perf_counter() - t0
    vv = sum(float((vi * vi).sum().item()) for vi in v_used)
    vw = sum(float((a * b).sum().item()) for a, b in zip(v_used, w))
    lam_ws = vw / max(vv, 1e-30)
    wn = math.sqrt(sum(float((wi * wi).sum().item()) for wi in w))
    if wn > 0:
        v_new = [wi / wn for wi in w]
    else:  # degenerate (zero HVP) — keep the old direction
        v_new = [vi.clone() for vi in v_used]
    return float(lam_ws), v_new, float(t_hvp)


def measure_curvature(model, opt, Xeval, yeval, n_iter, warm_v=None, ws_state=None):
    """λ_max (warm-started power iter) + single-HVP Rayleigh proxies, ALL in
    eval mode on the deterministic batch (Xeval, yeval). g, Hg, HΔθ share the
    same surface as λ_max (oracle correctness fix #1/#2). Returns
    (rec_dict, v_top) where v_top is warm-started across calls.

    ws_state (optional, default None → unchanged behavior): dict
    {"v": list[Tensor]|None, "init_seed": int} carrying the one-HVP
    warm-started power-iteration baseline state v_ws across calls. When given,
    rec gains lambda_hat_ws / cos_ws_vtop / t_hvp_g_s / t_hvp_ws_s.
    """
    ce = lambda p, y: F.cross_entropy(p, y)
    params = _param_list(model)
    model.eval()
    try:
        lam, v, n_hvp = max_eigenvalue(
            model, ce, Xeval, yeval, n_iter=n_iter, return_vec=True,
            return_iters=True, v_init=warm_v,
        )
        rec = {"lambda_max": float(lam), "lambda_max_n_hvp": int(n_hvp)}
        hvp_t: dict[str, float] = {}
        timer = (lambda which, dt: hvp_t.__setitem__(which, dt)) if ws_state is not None else None
        rec.update({k: float(val) for k, val in
                    rayleigh_proxies(model, opt, ce, Xeval, yeval, v, params,
                                     hvp_timer=timer).items()})
        if ws_state is not None:
            rec["t_hvp_g_s"] = float(hvp_t.get("grad", float("nan")))
            # --- one-HVP warm-started power-iteration baseline (same batch) ---
            v_used = ws_state["v"]
            if v_used is None:  # first measurement: seeded random unit vector
                gtor = torch.Generator(device=params[0].device)
                gtor.manual_seed(int(ws_state["init_seed"]))
                v_used = [
                    torch.randn(p.shape, generator=gtor, device=p.device,
                                dtype=p.dtype).detach()
                    for p in params
                ]
                nrm = math.sqrt(sum(float((vi * vi).sum().item()) for vi in v_used))
                v_used = [vi / nrm for vi in v_used]
            lam_ws, v_new, t_ws = _warm_start_step(
                model, ce, Xeval, yeval, params, v_used)
            # alignment of the vector USED for this estimate vs current v_top
            vn = math.sqrt(sum(float((a * a).sum().item()) for a in v))
            un = math.sqrt(sum(float((a * a).sum().item()) for a in v_used))
            cos_ws = (sum(float((a * b).sum().item()) for a, b in zip(v_used, v))
                      / max(vn * un, 1e-30)) if vn > 0 else float("nan")
            rec["lambda_hat_ws"] = lam_ws
            rec["cos_ws_vtop"] = float(cos_ws)
            rec["t_hvp_ws_s"] = t_ws
            ws_state["v"] = v_new
    finally:
        model.train()
    return rec, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="cifar_rayleigh")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--sharp-every", type=int, default=50)
    ap.add_argument("--sharp-batch", type=int, default=256)
    ap.add_argument("--sharp-niter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-start-track", action="store_true",
                    help="record the one-HVP warm-started power-iteration "
                         "baseline λ̂_ws (default OFF; without it the run is "
                         "identical to the original script)")
    args = ap.parse_args()

    LR, MOM, BS = args.lr, args.mom, args.bs
    STEPS, SHARP_EVERY, SHARP_BATCH, SHARP_NITER, SEED = (
        args.steps, args.sharp_every, args.sharp_batch, args.sharp_niter, args.seed)
    RUN_DIR = Path("runs") / args.run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading CIFAR-10 ...", flush=True)
    Xtr, ytr, task, Xte, yte = load_cifar10(device=DEV)
    N = len(Xtr)
    rng_eval = np.random.default_rng(123)
    idx_eval = rng_eval.integers(0, N, SHARP_BATCH)
    Xeval, yeval = Xtr[idx_eval], ytr[idx_eval]

    model = build_model("resnet", widths=(16, 32, 64), blocks=(2, 2, 2), num_classes=10).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet: {n_params/1e6:.2f}M params, lr={LR} (2/η={2/LR:g})", flush=True)
    opt = SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=args.wd)
    register_param_names(opt, model)

    snaps = SnapshotManager(every=args.snap_every, ref_center=True)
    rows = []
    rng = np.random.default_rng(SEED)
    warm_v = None
    # one-HVP warm-start baseline state (dedicated RNG → no global RNG use)
    ws_state = {"v": None, "init_seed": SEED} if args.warm_start_track else None

    for step in range(STEPS):
        idx = rng.integers(0, N, BS)
        Xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xb), yb)
        loss.backward()
        opt.step()

        if step % SHARP_EVERY == 0 or step == STEPS - 1:
            curv, warm_v = measure_curvature(model, opt, Xeval, yeval, SHARP_NITER,
                                             warm_v=warm_v, ws_state=ws_state)
            with torch.no_grad():
                gn = group_norms(model)
                tr_loss = float(F.cross_entropy(model(Xeval), yeval))
                acc = eval_acc(model, Xte[:2000], yte[:2000])
            snaps.maybe_add(step, model)
            rows.append(dict(step=step, train_loss=float(loss.item()), eval_loss=tr_loss,
                              eval_acc=acc, **curv, **{f"norm_{k}": v for k, v in gn.items()}))
            ws_str = ""
            if ws_state is not None:
                ws_str = (f" | λ̂_ws {curv['lambda_hat_ws']:+.2f}"
                          f" cos_ws {abs(curv['cos_ws_vtop']):.3f}"
                          f" ({curv['t_hvp_ws_s']*1e3:.0f}ms)")
            print(
                f"step {step:4d} | loss {loss.item():.3f} acc {acc:.3f} | "
                f"λ_max {curv['lambda_max']:.2f} (n_hvp {curv['lambda_max_n_hvp']}) | "
                f"λ̂_g {curv['lambda_hat_grad']:+.2f} λ̂_Δθ {curv['lambda_hat_update']:+.2f} | "
                f"cos_g {abs(curv['cos_grad_vtop']):.3f} cos_Δθ {abs(curv['cos_update_vtop']):.3f}"
                f"{ws_str}",
                flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / "metrics.csv", index=False)
    snaps.save(RUN_DIR / "snapshots.npz")
    manifest = dict(model="resnet", n_params=n_params, lr=LR, momentum=MOM, bs=BS, steps=STEPS,
                    final_loss=float(df["train_loss"].iloc[-1]), final_acc=float(df["eval_acc"].iloc[-1]),
                    lambda_max_final=float(df["lambda_max"].iloc[-1]))
    if args.warm_start_track:
        manifest.update(warm_start_track=True, seed=SEED,
                        ws_init_seed=SEED, sharp_every=args.sharp_every)
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] -> {RUN_DIR}")
    print(f"λ_max: init={df['lambda_max'].iloc[0]:.2f} max={df['lambda_max'].max():.2f} "
          f"final={df['lambda_max'].iloc[-1]:.2f}  (2/η={2/LR})")
    print(f"final acc {df['eval_acc'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
