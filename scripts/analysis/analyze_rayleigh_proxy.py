"""Candidate-1 analysis: single-HVP Rayleigh quotients as λ_max proxies.

Characterizes λ̂_grad = gᵀHg/gᵀg and λ̂_update = ΔθᵀHΔθ/ΔθᵀΔθ (each 1 HVP) against
the true λ_max (warm-started power iteration), on the fresh MLP (plain GD, the
equivalence check where Δθ=−ηg ⇒ λ̂_g≡λ̂_Δθ) and CIFAR-ResNet (momentum+BN, the
headline regime where Δθ≠−ηg) runs.

Positioned vs CDAT (Roulet NeurIPS'24) / Kalra 2026 who use uᵀHu for step-size
tuning; here as a 1-HVP λ_max MONITOR with the |cos(·,v_top)| validity regime.

Core figure: alignment-colored scatter (Rayleigh ≤ λ_max, tight iff aligned) +
the Δθ-vs-g divergence under momentum + a detrended-corr bar vs the Tier-0 norm
proxies (same-run baselines).

Usage: python scripts/analysis/analyze_rayleigh_proxy.py
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

# mathtext labels for the bar-chart proxy names (the CSV keeps the unicode originals)
_PROXY_MATH = {
    "λ̂_grad": r"$\hat{\lambda}_{\mathrm{grad}}$",
    "λ̂_update": r"$\hat{\lambda}_{\Delta\theta}$",
    "‖·‖ bn_gamma": r"$\|\mathrm{BN}\,\gamma\|$",
    "‖·‖ conv_w": r"$\|W_{\mathrm{conv}}\|$",
    "‖·‖ fc_w": r"$\|W_{\mathrm{fc}}\|$",
}

# (run_name, csv_name, regime_label, norm_proxy_cols)
SOURCES = [
    ("mlp_rayleigh", "sharpness.csv", "MLP plain-GD (Δθ=−ηg, equivalence)", []),
    ("cifar_rayleigh", "metrics.csv", "CIFAR ResNet momentum+BN (headline)",
     ["norm_bn_gamma", "norm_conv_w", "norm_fc_w"]),
]


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float(a @ b / d) if d > 0 else float("nan")


def detrend(a, w=5):
    a = np.asarray(a, float)
    k = np.ones(w) / w
    return a - np.pad(np.convolve(a, k, "same"), (0, 0), mode="edge")


def load(run, csv):
    m = pd.read_csv(RUNS / run / csv)
    lam = m["lambda_max"].to_numpy(float)
    f = np.isfinite(lam) & (lam > 1e-6)
    m = m.loc[f].reset_index(drop=True)
    return m


def regime_stats(m, col, align_col, thr=0.7):
    """(whole detrend r, eos detrend r, eos mean-rel-error)."""
    lam = m["lambda_max"].to_numpy(float)
    v = m[col].to_numpy(float)
    whole = pear(detrend(v), detrend(lam))
    eos = np.abs(m[align_col].to_numpy(float)) > thr
    if eos.sum() > 5:
        ve, le = v[eos], lam[eos]
        eos_r = pear(detrend(ve), detrend(le))
        eos_err = float(np.mean(np.abs(ve - le) / le))
    else:
        eos_r, eos_err = float("nan"), float("nan")
    return whole, eos_r, eos_err


def main():
    setup()
    tables = {}
    for run, csv, label, norm_cols in SOURCES:
        m = load(run, csv)
        lam = m["lambda_max"].to_numpy(float)
        print(f"\n=== {label}  ({run}, n={len(m)}) ===")
        rows = []
        for col, al in [("lambda_hat_grad", "cos_grad_vtop"),
                        ("lambda_hat_update", "cos_update_vtop")]:
            w, er, ee = regime_stats(m, col, al)
            cg = np.abs(m[al].to_numpy(float))
            rows.append(dict(proxy=col.replace("lambda_hat_", "λ̂_"),
                             whole_det_r=w, eos_det_r=er, eos_relerr=ee,
                             mean_align=float(cg.mean())))
            print(f"  {col:17s} whole det-r={w:+.3f}  EoS det-r={er:+.3f}  "
                  f"EoS rel-err={ee:.3f}  mean|cos|={cg.mean():.3f}")
        # same-run Tier-0 norm baselines
        for c in norm_cols:
            if c not in m.columns:
                continue
            v = m[c].to_numpy(float)
            rows.append(dict(proxy=c.replace("norm_", "‖·‖ "), whole_det_r=pear(detrend(v), detrend(lam)),
                             eos_det_r=float("nan"), eos_relerr=float("nan"), mean_align=float("nan")))
            print(f"  {c:17s} (Tier-0 baseline) whole det-r={pear(detrend(v), detrend(lam)):+.3f}")
        tables[run] = (m, pd.DataFrame(rows))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # Row 0: trajectories
    for ax, (run, csv, label, _), ttl in zip(
        [axes[0, 0], axes[0, 1]],
        SOURCES[:2], [r"MLP (equivalence: $\hat{\lambda}_{\mathbf{g}} \equiv \hat{\lambda}_{\Delta\theta}$)",
                      r"CIFAR momentum ($\Delta\theta \neq -\eta\mathbf{g}$: they diverge)"]
    ):
        m = load(run, csv)
        s = m["step"].to_numpy(float)
        ax.plot(s, m["lambda_max"], "o-", ms=3, lw=1.0, color="k", label=r"$\lambda_{\max}$ (true, power iter)")
        ax.plot(s, m["lambda_hat_grad"], "--", lw=1.2, color="tab:red", label=r"$\hat{\lambda}_{\mathrm{grad}}$ (1 HVP)")
        ax.plot(s, m["lambda_hat_update"], ":", lw=1.4, color="tab:blue", label=r"$\hat{\lambda}_{\Delta\theta}$ (1 HVP)")
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("step"); ax.set_ylabel("sharpness")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    # alignment overlay panel
    mC = load("cifar_rayleigh", "metrics.csv")
    s = mC["step"].to_numpy(float)
    axes[0, 2].plot(s, np.abs(mC["cos_grad_vtop"]), "-", lw=1.3, color="tab:red", label=r"$|\cos(\mathbf{g}, v_{\mathrm{top}})|$")
    axes[0, 2].plot(s, np.abs(mC["cos_update_vtop"]), "-", lw=1.3, color="tab:blue", label=r"$|\cos(\Delta\theta, v_{\mathrm{top}})|$")
    axes[0, 2].axhline(0.7, color="0.6", lw=0.7, ls="--", label="EoS threshold")
    axes[0, 2].set_title(r"CIFAR — $v_{\mathrm{top}}$ alignment: gradient vs update", fontsize=10)
    axes[0, 2].set_xlabel("step"); axes[0, 2].set_ylabel(r"$|\cos|$"); axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    # Row 1: scatter (the core lower-bound-tightening visual)
    for ax, (run, csv, col, al, ttl, c) in zip(
        [axes[1, 0], axes[1, 1]],
        [("mlp_rayleigh", "sharpness.csv", "lambda_hat_grad", "cos_grad_vtop", r"$\hat{\lambda}_{\mathrm{grad}}$ vs $\lambda_{\max}$ (MLP)", "tab:red"),
         ("cifar_rayleigh", "metrics.csv", "lambda_hat_grad", "cos_grad_vtop", r"$\hat{\lambda}_{\mathrm{grad}}$ vs $\lambda_{\max}$ (CIFAR)", "tab:red")]
    ):
        m = load(run, csv)
        lam = m["lambda_max"].to_numpy(float); v = m[col].to_numpy(float)
        c_al = np.abs(m[al].to_numpy(float))
        sc = ax.scatter(lam, v, c=c_al, cmap="viridis", s=22, edgecolor="0.3", lw=0.3, vmin=0, vmax=1)
        hi = max(lam.max(), v.max()) * 1.05
        ax.plot([0, hi], [0, hi], "k--", lw=0.8, label=r"$y=x$ ($\hat{\lambda}=\lambda_{\max}$)")
        ax.set_title(ttl + f"  (EoS det-r={regime_stats(m,col,al)[1]:+.2f})", fontsize=10)
        ax.set_xlabel(r"$\lambda_{\max}$ (true)"); ax.set_ylabel(r"$\hat{\lambda}_{\mathrm{grad}}$ (1 HVP)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label=r"$|\cos(\mathbf{g}, v_{\mathrm{top}})|$")
    # Δθ-vs-g comparison + detrended-r bar
    mC = load("cifar_rayleigh", "metrics.csv")
    axes[1, 2].scatter(mC["lambda_hat_grad"], mC["lambda_hat_update"],
                       c=np.abs(mC["cos_grad_vtop"]), cmap="viridis", s=22, edgecolor="0.3", lw=0.3)
    mx = max(mC["lambda_hat_grad"].max(), mC["lambda_hat_update"].max()) * 1.05
    axes[1, 2].plot([0, mx], [0, mx], "k--", lw=0.8)
    axes[1, 2].set_title(r"CIFAR: $\hat{\lambda}_{\Delta\theta}$ vs $\hat{\lambda}_{\mathrm{grad}}$" + "\n" +
                         r"(point below diagonal $\Rightarrow$ update is the WORSE proxy)", fontsize=9)
    axes[1, 2].set_xlabel(r"$\hat{\lambda}_{\mathrm{grad}}$"); axes[1, 2].set_ylabel(r"$\hat{\lambda}_{\Delta\theta}$"); axes[1, 2].grid(alpha=0.3)

    fig.suptitle(r"Candidate 1 — single-HVP Rayleigh quotients as $\lambda_{\max}$ proxy (vs CDAT/Kalra; "
                 r"regime: Rayleigh $\leq$ $\lambda_{\max}$, tight iff $|\cos(\cdot, v_{\mathrm{top}})| \to 1$)", y=1.00, fontsize=12)
    fig.tight_layout()
    out = savefig(fig, "23_rayleigh_proxy")
    print(f"\nsaved {out}")

    # bar chart of CIFAR detrended-r (proxies vs Tier-0 norms)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dfC = tables["cifar_rayleigh"][1]
    dfC = dfC.sort_values("whole_det_r", ascending=True, key=lambda s: s.abs())
    cols = ["tab:red" if "grad" in p else ("tab:blue" if "update" in p else "0.6") for p in dfC["proxy"]]
    ax.barh(dfC["proxy"], dfC["whole_det_r"], color=cols)
    ax.set_yticks(range(len(dfC)))
    ax.set_yticklabels([_PROXY_MATH.get(p, p) for p in dfC["proxy"]])
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_xlabel(r"detrended Pearson corr with $\lambda_{\max}$ (whole trajectory, CIFAR)")
    ax.set_title("Candidate 1 — CIFAR: single-HVP proxies vs Tier-0 norm baselines")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); out2 = savefig(fig, "23b_rayleigh_proxy_corrbar")
    print(f"saved {out2}")
    dfC.to_csv(FIG / "rayleigh_proxy_summary.csv", index=False)


if __name__ == "__main__":
    main()
