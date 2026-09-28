"""Gap-E alignment analysis on timm models (Finding 11 generalization).

For ViT-tiny and ConvNeXt-atto, 4 seeds each (CIFAR-10): is each run's weight
trajectory still ~1-D, and are the 1-D directions still near-orthogonal across
seeds (MLP's 'no common skeleton', Finding 11) — or do modern architectures
align? Uses power-iteration PCA (low memory: W + a few vectors) so it fits a
contended GPU and falls back to CPU.

Run (background):  nohup/tmux — writes figures/20_timm_multiseed_gape.png +
figures/timm_multiseed_alignment.csv + prints the summary.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def top2_pc(Wc: torch.Tensor, iters: int = 35):
    """Top-2 right singular vectors/values via orthogonal (power) iteration.
    Memory peak ≈ ‖Wc‖ + a couple of P-vectors (no full SVD)."""
    P = Wc.shape[1]
    def power(v):
        v = v / (v.norm() + 1e-30)
        for _ in range(iters):
            u = Wc @ v
            v = Wc.t() @ u
            n = v.norm()
            if n < 1e-12:
                break
            v = v / n
        s2 = float((Wc @ v).pow(2).sum())
        return v, s2
    v1, s1 = power(torch.randn(P, device=Wc.device, dtype=Wc.dtype))
    v2 = torch.randn(P, device=Wc.device, dtype=Wc.dtype)
    v2 = v2 - float(v1 @ v2) * v1
    v2 = v2 / (v2.norm() + 1e-30)
    for _ in range(iters):
        u = Wc @ v2
        v2 = Wc.t() @ u
        v2 = v2 - float(v1 @ v2) * v1
        n = v2.norm()
        if n < 1e-12:
            break
        v2 = v2 / n
    s2 = float((Wc @ v2).pow(2).sum())
    return (v1, s1), (v2, s2)


def analyze_arch(arch: str, n_seeds: int = 4):
    V1, o1, o12 = [], [], []
    for s in range(n_seeds):
        p = Path(f"runs/multiseed_{arch}/seed_{s}/snapshots.npz")
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        W = torch.from_numpy(z["W"].astype(np.float32)).to(DEV)
        Wc = W - W.mean(0)
        fro2 = float(Wc.pow(2).sum())
        (v1, s1), (v2, s2) = top2_pc(Wc)
        o1.append(s1 / fro2); o12.append((s1 + s2) / fro2)
        V1.append(v1.double().cpu().numpy())
        del W, Wc
        if DEV == "cuda":
            torch.cuda.empty_cache()
        print(f"  {arch} seed {s}: own_pc1={o1[-1]*100:.0f}% own_pc1+2={o12[-1]*100:.0f}%", flush=True)
    n = len(V1)
    M = np.array([[abs(float(V1[i] @ V1[j])) for j in range(n)] for i in range(n)]) if n else np.zeros((0, 0))
    off = (M.sum() - n) / (n * (n - 1)) if n > 1 else float("nan")
    return dict(arch=arch, own_pc1=float(np.mean(o1)), own_pc12=float(np.mean(o12)),
                pairwise_cos=float(off), V1=V1)


def main():
    print(f"device = {DEV}", flush=True)
    out = [analyze_arch(a) for a in ["timm_vit_tiny", "timm_convnext_atto"]]
    print("\n=== Gap-E on timm (Finding 11 generalization) ===", flush=True)
    print(f"{'arch':22s} | own_pc1 | own_pc1+2 | mean pairwise |cos(PC1)|")
    for r in out:
        print(f"{r['arch']:22s} | {r['own_pc1']*100:.0f}%    | {r['own_pc12']*100:.0f}%      | {r['pairwise_cos']:.3f}")
    print("MLP (Finding 11) ref : 94%     | 98%      | 0.24  => 1-D each, near-orthogonal => NO common skeleton", flush=True)
    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items() if k != "V1"} for r in out]).to_csv(
        "figures/timm_multiseed_alignment.csv", index=False)

    # figure: pairwise |cos| heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, r in zip(axes, out):
        V1 = r["V1"]; n = len(V1)
        M = np.array([[abs(float(V1[i] @ V1[j])) for j in range(n)] for i in range(n)])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"{r['arch']}\nmean off-diag |cos(PC1)|={r['pairwise_cos']:.2f}  (low ⇒ no common skeleton)")
        ax.set_xlabel("seed"); ax.set_ylabel("seed")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Finding 17? — multi-seed trajectory alignment on timm (ViT / ConvNeXt, LayerNorm)", y=1.02)
    fig.tight_layout(); Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/20_timm_multiseed_gape.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nDONE -> figures/20_timm_multiseed_gape.png + figures/timm_multiseed_alignment.csv", flush=True)


if __name__ == "__main__":
    main()
