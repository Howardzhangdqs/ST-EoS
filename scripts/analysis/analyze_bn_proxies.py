"""Finding 13 follow-up: which free per-step proxy predicts λ_max under BatchNorm?

Finding 9 (tanh-MLP): ‖last-hidden W⟩ correlates +0.8 with λ_max.
Finding 13 (BN convnet): that proxy flips to −0.8. So under BN, what (if anything)
tracks sharpness? We compute the Pearson correlation of every parameter-group
norm with λ_max along the CIFAR ResNet trajectory.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUN = Path("runs/cifar_resnet_eos")
FIG = Path("figures")

NICE = dict(norm_conv_w="‖conv W‖", norm_bn_gamma="‖BN γ‖",
            norm_bn_beta="‖BN β‖", norm_fc_w="‖fc W‖", norm_other_w="‖other W‖")


def main():
    m = pd.read_csv(RUN / "metrics.csv")
    lam = m["lambda_max"].to_numpy(float)
    finite = np.isfinite(lam) & (lam > 0)
    proxy_cols = [c for c in m.columns if c.startswith("norm_")]

    def pear(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        am, bm = a.mean(), b.mean()
        den = np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum())
        return float(((a - am) @ (b - bm)) / den) if den > 0 else float("nan")

    def detrend(a, w=5):
        a = np.asarray(a, float)
        if len(a) < 2 * w:
            return a - a.mean()
        # rolling-mean residual (high-pass): removes the monotone global drift
        k = np.ones(w) / w
        pad = w // 2
        trend = np.pad(np.convolve(a, k, mode="same"), (0, 0), mode="edge")
        return a - trend

    rows = []
    for c in proxy_cols:
        v = m[c].to_numpy(float)[finite]
        l = lam[finite]
        raw = pear(v, l)
        det_proxy = detrend(v); det_lam = detrend(l)
        det = pear(det_proxy, det_lam)
        # first-difference corr (step-to-step co-movement)
        dproxy = np.diff(v); dlam = np.diff(l)
        dif = pear(dproxy, dlam)
        rows.append(dict(proxy=NICE.get(c, c), col=c, raw_pearson=raw,
                         detrended_pearson=det, diff_pearson=dif))
    df = pd.DataFrame(rows).sort_values("detrended_pearson", ascending=False, key=lambda s: s.abs())
    print("\n=== BN sharpness-proxy candidates vs λ_max ===")
    print("  (raw is trend-confounded; DETRENDED and DIFF are the meaningful signals)")
    print(df[["proxy", "raw_pearson", "detrended_pearson", "diff_pearson"]].to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"))

    best = df.iloc[0]
    print(f"\n=> best detrended free proxy under BN: {best['proxy']}  "
          f"(detrended r={best['detrended_pearson']:+.3f}, diff r={best['diff_pearson']:+.3f})")

    # figure: detrended λ_max vs each detrended proxy + detrended-corr bar
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    l = lam[finite]; steps = m["step"].to_numpy()[finite]
    axes[0].plot(steps, detrend(l), "-", lw=1.4, color="black", label="Δλ_max (detrended)")
    colors = dict(norm_conv_w="tab:blue", norm_bn_gamma="tab:red", norm_bn_beta="tab:purple",
                  norm_fc_w="tab:green", norm_other_w="tab:gray", norm_other="tab:orange")
    for c in proxy_cols:
        v = m[c].to_numpy(float)[finite]
        axes[0].plot(steps, detrend(v) / max(np.std(detrend(v)), 1e-9), "--", lw=0.8, alpha=0.75,
                     color=colors.get(c, "0.5"),
                     label=f"{NICE.get(c,c)} (det-r={df[df.col==c]['detrended_pearson'].iloc[0]:+.2f})")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("detrended (unit-std)")
    axes[0].set_title("Detrended: does any proxy track λ_max's short-term motion?")
    axes[0].legend(fontsize=7, loc="best"); axes[0].grid(alpha=0.3)

    order = df.sort_values("detrended_pearson").index
    axes[1].barh(range(len(df)), df.loc[order, "detrended_pearson"],
                 color=[colors.get(c, "0.6") for c in df.loc[order, "col"]])
    axes[1].set_yticks(range(len(df))); axes[1].set_yticklabels(df.loc[order, "proxy"].tolist())
    axes[1].axvline(0, color="0.6", lw=0.6); axes[1].set_xlabel("detrended Pearson corr with λ_max")
    axes[1].set_title("Short-term proxy fidelity under BN"); axes[1].grid(alpha=0.3, axis="x")
    fig.suptitle("Finding 13 follow-up — detrended sharpness-proxy search under BatchNorm", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "17_bn_sharpness_proxies.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    df.to_csv(FIG / "bn_proxy_correlations.csv", index=False)
    print("saved figures/17_bn_sharpness_proxies.png + bn_proxy_correlations.csv")


if __name__ == "__main__":
    main()
