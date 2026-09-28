"""Multi-run analysis of the timm sweep: ViT-tiny + ConvNeXt-{atto,femto} on
CIFAR-10/100. For each run: EoS summary, trajectory low-dimensionality (PCA),
and the detrended correlation of every parameter-group norm with λ_max — the
Finding-14 generalization test (does LN-γ win under LayerNorm, as BN-γ won
under BatchNorm?).
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.multiseed import _pca_dual

RUNS = Path("runs")
FIG = Path("figures")
CONDITIONS = [
    ("timm_vit_tiny", "cifar10"), ("timm_vit_tiny", "cifar100"),
    ("timm_convnext_atto", "cifar10"), ("timm_convnext_atto", "cifar100"),
    ("timm_convnext_femto", "cifar10"), ("timm_convnext_femto", "cifar100"),
]
NICE = dict(ln_gamma="‖LN γ‖", ln_beta="‖LN β‖", bn_gamma="‖BN γ‖",
            conv_w="‖conv W‖", lin_w="‖lin W‖")


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    am, bm = a.mean(), b.mean(); den = np.sqrt(((a-am)**2).sum()*((b-bm)**2).sum())
    return float(((a-am)@(b-bm))/den) if den > 0 else float("nan")


def detrend(a, w=5):
    a = np.asarray(a, float)
    if len(a) < 2*w: return a - a.mean()
    k = np.ones(w)/w; trend = np.pad(np.convolve(a, k, "same"), (0, 0), mode="edge")
    return a - trend


def analyze(run_name):
    m = pd.read_csv(RUNS / run_name / "metrics.csv")
    z = np.load(RUNS / run_name / "snapshots.npz", allow_pickle=True)
    W = z["W"].astype(np.float64)
    _, expl = _pca_dual(W - W.mean(0))
    lam = m["lambda_max"].to_numpy(float)
    fin = np.isfinite(lam) & (lam > 0)
    proxy_cols = [c for c in m.columns if c.startswith("norm_")]
    best_name, best_r = None, 0.0
    det_by_proxy = {}
    for c in proxy_cols:
        v = m[c].to_numpy(float)[fin]; l = lam[fin]
        r = pear(detrend(v), detrend(l))
        base = c.replace("norm_", "")
        det_by_proxy[base] = r
        if abs(r) > abs(best_r):
            best_r = r; best_name = base
    return dict(run=run_name, acc=float(m["eval_acc"].iloc[-1]),
                lambda_max_init=float(lam[fin][0]), lambda_max_max=float(lam[fin].max()),
                lambda_max_final=float(lam[fin][-1]),
                pc1=float(expl[0]), pc12=float(expl[:2].sum()),
                best_proxy=best_name, best_proxy_det_r=best_r, **{f"det_{k}": v for k, v in det_by_proxy.items()})


def main():
    rows = [analyze(f"{mo}__{ds}") for mo, ds in CONDITIONS]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\n=== timm sweep summary ===")
    cols = ["run", "acc", "lambda_max_init", "lambda_max_max", "lambda_max_final", "pc1", "pc12", "best_proxy", "best_proxy_det_r"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\n=== detrended corr of each proxy with λ_max (the Finding-14 generalization test) ===")
    det_cols = [c for c in df.columns if c.startswith("det_")]
    print(df[["run"] + det_cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    df.to_csv(FIG / "timm_sweep_summary.csv", index=False)

    # figure: 3×2 grid, each panel = detrended λ_max vs detrended ‖LN γ‖ (best proxy highlighted)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (mo, ds) in zip(axes.flat, CONDITIONS):
        run = f"{mo}__{ds}"
        m = pd.read_csv(RUNS / run / "metrics.csv")
        lam = m["lambda_max"].to_numpy(float)
        z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        ax.plot(m["step"], z(detrend(lam)), "-", lw=1.4, color="black", label="Δλ_max")
        best = df[df.run == run]["best_proxy"].iloc[0]
        for c in [c for c in m.columns if c.startswith("norm_")]:
            base = c.replace("norm_", "")
            v = m[c].to_numpy(float)
            col = "tab:red" if base == best else ("0.7")
            lw = 1.3 if base == best else 0.7
            lab = NICE.get(base, base) if base == best else None
            ax.plot(m["step"], z(detrend(v)), "--", lw=lw, alpha=0.9 if base == best else 0.5,
                    color=col, label=lab)
        short = mo.replace("timm_", "") + "/" + ds
        ax.set_title(f"{short}  (best proxy: {NICE.get(best,best)}, det-r={df[df.run==run]['best_proxy_det_r'].iloc[0]:+.2f})", fontsize=9)
        ax.set_xlabel("step", fontsize=8); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    fig.suptitle("Finding 16 — sharpness-proxy search across ViT & ConvNeXt (LayerNorm)", y=1.01)
    fig.tight_layout(); fig.savefig(FIG / "19_timm_sweep_proxies.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nsaved figures/19_timm_sweep_proxies.png + timm_sweep_summary.csv")


if __name__ == "__main__":
    main()
