"""Generalization analysis for the CIFAR ResNet EoS run.
Tests whether Findings 3 (low-dim trajectory) and 9 (weight-norm proxy)
survive the move to a normalized convnet on real images.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.multiseed import _pca_dual

RUN = Path("runs/cifar_resnet_eos")
FIG = Path("figures")


def main():
    m = pd.read_csv(RUN / "metrics.csv")
    z = np.load(RUN / "snapshots.npz", allow_pickle=True)
    W = z["W"].astype(np.float64)
    Vt, expl = _pca_dual(W - W.mean(0))
    print(f"trajectory PCA: PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  PC1+2={expl[:2].sum()*100:.1f}%")

    # Gap B on convnet: corr(layer norm, lambda_max) along trajectory
    lam = m["lambda_max"].to_numpy(float)
    mask = np.isfinite(lam) & (lam > 0)
    def corr(c):
        v = m[c].to_numpy(float)[mask]; l = lam[mask]
        return float(np.corrcoef(v, l)[0, 1])
    print(f"corr(‖fc.weight‖, λ_max)     = {corr('wnorm_fc'):.3f}")
    print(f"corr(‖lastconv.weight‖, λ_max) = {corr('wnorm_lastconv'):.3f}")
    print(f"λ_max range along trajectory: {lam[mask].min():.1f} … {lam[mask].max():.1f}  (2/η=20; momentum μ=0.9 lifts the effective threshold)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(m["step"], m["eval_loss"], "-", lw=0.9, color="tab:blue", label="eval loss")
    axb = axes[0].twinx(); axb.plot(m["step"], m["eval_acc"], "-", lw=0.9, color="tab:cyan", label="eval acc")
    axes[0].set_title("CIFAR-10 ResNet (BN)"); axes[0].set_ylabel("eval loss"); axb.set_ylabel("acc")
    axes[0].set_xlabel("step"); axes[0].set_yscale("log")

    axes[1].plot(m["step"], m["lambda_max"], "o-", ms=3, lw=0.7, color="tab:purple")
    axes[1].axhline(2/0.1, color="red", ls="--", lw=1, label="2/η=20 (vanilla-GD threshold)")
    axes[1].axhline(76, color="0.5", ls=":", lw=1, label="~observed momentum-EoS level")
    axes[1].set_ylabel("λ_max"); axes[1].set_yscale("log"); axes[1].set_xlabel("step")
    axes[1].set_title("sharpness (momentum-SGD EoS)"); axes[1].legend(fontsize=7)

    axes[2].plot(m["step"], m["wnorm_fc"], "-", lw=0.9, color="tab:green", label="‖fc‖")
    axes[2].plot(m["step"], m["wnorm_lastconv"], "-", lw=0.9, color="tab:orange", label="‖last_conv‖")
    axes[2].set_ylabel("weight norm"); axes[2].set_xlabel("step")
    axes[2].set_title(f"layer norms (proxy) — corr with λ_max: fc={corr('wnorm_fc'):.2f}, conv={corr('wnorm_lastconv'):.2f}")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]: ax.grid(alpha=0.3)
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"Generalization to CIFAR-10 + PreAct-ResNet(BN) — trajectory PC1={expl[0]*100:.0f}%, PC1+2={expl[:2].sum()*100:.0f}%", y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "16_cifar_resnet_generalization.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved figures/16_cifar_resnet_generalization.png")


if __name__ == "__main__":
    main()
