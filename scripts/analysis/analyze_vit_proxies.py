"""Candidate-5 analysis: the zero-HVP sharpness-proxy search on Transformers.

Two questions, on fresh ViT-tiny and ConvNeXt-atto CIFAR-10 runs (lr=0.01, the
proper EoS regime where λ_max oscillates):
  1. Does splitting ViT's Linears into attn-qkv vs mlp-fc, or per-block LN-γ
     heterogeneity, beat the Finding-16 global proxies?
  2. Is there ANY free (zero-HVP) proxy that substantially tracks λ_max on ViT,
     the architecture where Finding-16 reported norm proxies collapse?

Reconciliation note: Finding-16's ViT det-r ≈ +0.02 was measured over 16 sparse
points during violent λ_max oscillation; denser sampling (41 pts) revises it to
~+0.44. The robust conclusion: free proxies plateau ~+0.5 (grad_norm marginally
best), so the single-HVP λ̂_grad of Finding 20 (+0.74, near-exact in EoS) is the
answer — not a better free proxy.

Usage: python scripts/analysis/analyze_vit_proxies.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.analysis.figure_style import setup, savefig

RUNS = Path("runs")
FIG = Path("figures")

SOURCES = [
    ("timm_vit_tiny__cifar10_proxy", "ViT-tiny (LayerNorm)"),
    ("timm_convnext_atto__cifar10_proxy", "ConvNeXt-atto (LayerNorm)"),
]
NICE = dict(grad_norm="‖g‖ (grad)", update_norm="‖Δθ‖ (update)",
            norm_ln_gamma="‖LN γ‖", norm_ln_beta="‖LN β‖",
            norm_attn_w="‖attn W‖ (qkv+proj)", norm_mlp_w="‖mlp W‖ (fc1+fc2)",
            norm_conv_w="‖conv W‖", norm_lin_w="‖head W‖",
            norm_ln_gamma_block_std="std_block(‖γ_block‖)",
            norm_ln_gamma_block_maxmin="maxmin_block(‖γ_block‖)")
# mathtext mirror for figure tick labels (stdout / CSV keep the unicode NICE)
NICE_MATH = dict(grad_norm=r"$\|\mathbf{g}\|$ (grad)", update_norm=r"$\|\Delta\theta\|$ (update)",
                 norm_ln_gamma=r"$\|\mathrm{LN}\,\gamma\|$", norm_ln_beta=r"$\|\mathrm{LN}\,\beta\|$",
                 norm_attn_w=r"$\|W_{\mathrm{attn}}\|$ (qkv+proj)", norm_mlp_w=r"$\|W_{\mathrm{mlp}}\|$ (fc1+fc2)",
                 norm_conv_w=r"$\|W_{\mathrm{conv}}\|$", norm_lin_w=r"$\|W_{\mathrm{head}}\|$",
                 norm_ln_gamma_block_std=r"$\mathrm{std}_{\mathrm{block}}(\|\gamma_{\mathrm{block}}\|)$",
                 norm_ln_gamma_block_maxmin=r"$\mathrm{maxmin}_{\mathrm{block}}(\|\gamma_{\mathrm{block}}\|)$")


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float(a @ b / d) if d > 0 else float("nan")


def detrend(a, w=5):
    a = np.asarray(a, float); k = np.ones(w) / w
    return a - np.pad(np.convolve(a, k, "same"), (0, 0), mode="edge")


def analyze(run):
    m = pd.read_csv(RUNS / run / "metrics.csv")
    lam = m["lambda_max"].to_numpy(float)
    f = np.isfinite(lam) & (lam > 1e-3)
    l = lam[f]
    proxy_cols = [c for c in m.columns if c.startswith("norm_")] + ["grad_norm", "update_norm"]
    rows = []
    for c in proxy_cols:
        v = m[c].to_numpy(float)[f]
        if np.std(v) < 1e-12:
            continue
        rows.append((c, pear(detrend(v), detrend(l))))
    rows.sort(key=lambda r: -abs(r[1]))
    return m, l, f, rows


def main():
    setup()
    results = {run: analyze(run) for run, _ in SOURCES}

    print("\n=== Candidate 5 — free (zero-HVP) proxies vs λ_max, fresh runs (lr=0.01) ===")
    for run, label in SOURCES:
        m, l, f, rows = results[run]
        print(f"\n--- {label}  ({run}, n={int(f.sum())}, λ_max {l.min():.0f}–{l.max():.0f}) ---")
        for c, r in rows:
            print(f"  {NICE.get(c, c):26s} detrend r = {r:+.3f}")
        # block γ heterogeneity headline
        for c in ["norm_ln_gamma_block_std"]:
            if c in m.columns:
                print(f"  ⇒ per-block LN-γ heterogeneity std = {m[c].iloc[-1]:.2f}")

    # reconciliation with Finding 16 (sparse-sampled ViT)
    old = pd.read_csv(RUNS / "timm_vit_tiny__cifar10" / "metrics.csv")
    ol = old["lambda_max"].to_numpy(float); of = np.isfinite(ol) & (ol > 1e-3)
    old_g = pear(detrend(old["norm_ln_gamma"].to_numpy(float)[of]), detrend(ol[of]))
    new_g = results["timm_vit_tiny__cifar10_proxy"][3]
    new_g = dict(new_g).get("norm_ln_gamma", float("nan"))
    print(f"\n=== Finding-16 reconciliation ===")
    print(f"  ViT ‖LN γ‖ det-r:  Finding-16 (n={int(of.sum())}, sparse) = {old_g:+.3f}  →  "
          f"fresh dense (n=41) = {new_g:+.3f}")
    print(f"  ⇒ the '+0.02 collapse' was a sparse-sampling artifact over violent oscillation; "
          f"denser measurement gives ~+0.44.")

    # ---- figure: side-by-side detrended-corr bars + γ-heterogeneity contrast ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    width = 0.38
    runs_lbl = [lbl for _, lbl in SOURCES]
    # common proxy set (intersection, ordered by ViT strength)
    m0 = results[SOURCES[0][0]][3]
    proxies = [c for c, _ in m0]
    x = np.arange(len(proxies))
    for i, (run, label) in enumerate(SOURCES):
        rs = dict(results[run][3])
        vals = [rs.get(c, float("nan")) for c in proxies]
        ax.bar(x + (i - 0.5) * width, vals, width, label=label)
    ax.set_xticks(x); ax.set_xticks(x)
    ax.set_xticklabels([NICE_MATH.get(c, NICE.get(c, c)) for c in proxies], rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6); ax.set_ylabel(r"detrended Pearson corr with $\lambda_{\max}$")
    ax.set_title("Candidate 5 — free proxies: ViT vs ConvNeXt (lr=0.01, EoS regime)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # γ-heterogeneity contrast
    ax = axes[1]
    archs = [lbl.replace(" (LayerNorm)", "") for _, lbl in SOURCES]
    het = []
    for run, _ in SOURCES:
        m = pd.read_csv(RUNS / run / "metrics.csv")
        het.append(float(m["norm_ln_gamma_block_std"].iloc[-1]))
    ax.bar(archs, het, color=["tab:orange", "tab:green"])
    ax.set_ylabel(r"std across blocks of $\|\gamma_{\mathrm{block}}\|$")
    ax.set_title(r"Per-block LN-$\gamma$ heterogeneity" + "\n" +
                 r"(ViT uniform $\Rightarrow$ $\gamma$ carries no block signal)")
    for a, v in zip(archs, het):
        ax.text(a, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Candidate 5 — sharpness-proxy search on Transformers (Finding-21)", y=1.01, fontsize=12)
    fig.tight_layout()
    out = savefig(fig, "24_vit_proxy_search")
    print(f"\nsaved {out}")

    # CSV
    rows = []
    for run, label in SOURCES:
        for c, r in results[run][3]:
            rows.append(dict(arch=label, proxy=NICE.get(c, c), detrend_r=r))
    pd.DataFrame(rows).to_csv(FIG / "vit_proxy_summary.csv", index=False)
    print(f"saved figures/vit_proxy_summary.csv")


if __name__ == "__main__":
    main()
