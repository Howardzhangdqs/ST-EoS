"""Generic EoS-on-CIFAR runner for timm models (ViT, ConvNeXt, …).

Records λ_max (eval mode, deterministic for normalization), the trajectory
snapshots (for low-dim PCA), and — by MODULE TYPE — the combined norm of each
parameter group (LN-γ, LN-β, BN-γ, BN-β, conv W, linear W). The LayerNorm γ is
the Finding-14 generalization candidate: is the normalization affine scale the
sharpness proxy across normalization families (BN → LN)?

Usage:
    python scripts/experiments/train_timm.py --model timm_vit_tiny --dataset cifar10 --lr 0.01
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data.cifar import load_cifar
from steos.models import build_model
from steos.optim import SGD, register_param_names
from steos.sharpness import max_eigenvalue
from steos.monitor.snapshots import SnapshotManager

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def group_norms(model):
    """Per-module-TYPE parameter norms — the sharpness-proxy search space.

    Finding-16 split: timm ViT has 24 attn-Linears (qkv+proj) + 24 mlp-Linears
    (fc1+fc2) + head; ConvNeXt is all-conv (no internal Linears except head) —
    which is exactly why LN γ wins on ConvNeXt (+0.57) but fails on ViT (+0.02):
    ConvNeXt's curvature-bearing params ARE the conv/LN chain; ViT's are the
    attn-qkv Linears downstream of LN. We therefore split Linears by path
    (attn / mlp / head) and add per-block LN-γ heterogeneity (std across blocks),
    since curvature localizes block-wise (Finding 9/11).
    """
    import re
    g: dict[str, list[torch.Tensor]] = defaultdict(list)
    block_gamma: dict[int, list[torch.Tensor]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            g["bn_gamma"].append(mod.weight.detach().reshape(-1))
            if mod.bias is not None:
                g["bn_beta"].append(mod.bias.detach().reshape(-1))
        elif isinstance(mod, nn.LayerNorm):
            g["ln_gamma"].append(mod.weight.detach().reshape(-1))
            if mod.bias is not None:
                g["ln_beta"].append(mod.bias.detach().reshape(-1))
            # per-block grouping: "...blocks.<i>.norm..."
            m = re.search(r"blocks\.(\d+)\.", name)
            if m:
                block_gamma.setdefault(int(m.group(1)), []).append(mod.weight.detach().reshape(-1))
        elif isinstance(mod, nn.Conv2d):
            g["conv_w"].append(mod.weight.detach().reshape(-1))
        elif isinstance(mod, nn.Linear):
            tag = "attn_w" if ("attn" in name or "attention" in name) else (
                  "mlp_w" if "mlp" in name else "lin_w")
            g[tag].append(mod.weight.detach().reshape(-1))
    out = {k: float(torch.cat(v).norm().item()) for k, v in g.items() if v}
    # per-block LN-γ heterogeneity (ViT: 12 blocks × {norm1,norm2})
    if block_gamma:
        per_block = [float(torch.cat(v).norm().item()) for v in block_gamma.values()]
        out["ln_gamma_block_mean"] = float(np.mean(per_block))
        out["ln_gamma_block_std"] = float(np.std(per_block))
        out["ln_gamma_block_maxmin"] = float(max(per_block) - min(per_block))
    return out


@torch.no_grad()
def eval_acc(model, Xv, yv, bs=1000):
    model.eval(); c = 0
    for i in range(0, len(Xv), bs):
        c += int((model(Xv[i:i+bs]).argmax(1) == yv[i:i+bs]).sum())
    model.train(); return c / len(Xv)


def measure_sharpness(model, Xb, yb, n_iter):
    # Force the SDPA math backend — the flash/efficient kernels lack the
    # second-order derivative the Hessian-vector product (double-backward) needs.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    model.eval()
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            lam = float(max_eigenvalue(model, lambda p, y: F.cross_entropy(p, y), Xb, yb, n_iter=n_iter))
    finally:
        model.train()
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--sharp-batch", type=int, default=96)
    ap.add_argument("--sharp-every", type=int, default=50)
    ap.add_argument("--sharp-niter", type=int, default=60)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    n_classes = 100 if args.dataset == "cifar100" else 10
    run_name = args.run_name or f"{args.model}_{args.dataset}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Xtr, ytr, task, Xte, yte = load_cifar(n_classes=n_classes, device=DEV)
    N = len(Xtr)
    idx_e = np.random.default_rng(123).integers(0, N, args.sharp_batch)
    Xe, ye = Xtr[idx_e], ytr[idx_e]

    model = build_model(args.model, num_classes=n_classes).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_name}] {args.model} on {args.dataset}: {n_params/1e6:.2f}M params, lr={args.lr}", flush=True)
    opt = SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    register_param_names(opt, model)

    snaps = SnapshotManager(every=args.snap_every, ref_center=True)
    rows = []; rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.bs); Xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(Xb), yb); loss.backward()
        with torch.no_grad():
            grad_norm = float(np.sqrt(sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)))
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        with torch.no_grad():
            update_norm = float(np.sqrt(sum((p - before[n]).pow(2).sum().item()
                                            for n, p in model.named_parameters())))
        snaps.maybe_add(step, model)  # unconditional — gated by snap_every internally
        if step % args.sharp_every == 0 or step == args.steps - 1:
            lam = measure_sharpness(model, Xe, ye, args.sharp_niter)
            with torch.no_grad():
                gn = group_norms(model)
                tr_loss = float(F.cross_entropy(model(Xe), ye)); acc = eval_acc(model, Xte[:2000], yte[:2000])
            rows.append(dict(step=step, train_loss=float(loss.item()), eval_loss=tr_loss, eval_acc=acc,
                              lambda_max=lam, grad_norm=grad_norm, update_norm=update_norm,
                              **{f"norm_{k}": v for k, v in gn.items()}))
            print(f"  step {step:4d} | loss {loss.item():.3f} acc {acc:.3f} | λ_max {lam:.1f} | "
                  + " ".join(f"{k}={v:.2f}" for k, v in gn.items()), flush=True)

    df = pd.DataFrame(rows); df.to_csv(run_dir / "metrics.csv", index=False)
    snaps.save(run_dir / "snapshots.npz")
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        model=args.model, dataset=args.dataset, n_params=n_params, lr=args.lr, mom=args.mom, wd=args.wd,
        bs=args.bs, steps=args.steps, final_loss=float(df["train_loss"].iloc[-1]),
        final_acc=float(df["eval_acc"].iloc[-1]), lambda_max_final=float(df["lambda_max"].iloc[-1]),
        lambda_max_max=float(df["lambda_max"].max())), indent=2))
    print(f"[{run_name}] DONE | λ_max init {df['lambda_max'].iloc[0]:.1f} max {df['lambda_max'].max():.1f} "
          f"final {df['lambda_max'].iloc[-1]:.1f} | acc {df['eval_acc'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
