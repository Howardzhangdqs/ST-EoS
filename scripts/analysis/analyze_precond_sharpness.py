"""Preconditioned (effective) sharpness analysis — review round 2, theory item.

The reviewer's reformulation: a gradient edit that rescales the component
along the top eigendirection by a factor s (damping: s = 1/d; amplification:
s = 1 + α) changes the stability-governing quantity from the RAW sharpness
λ1 to the PRECONDITIONED sharpness λ_eff = s·λ1. If the EoS mechanism
regulates λ_eff to the boundary 2/η, then the raw λ1 is free to sit at
2/(sη) — damping RAISES the raw equilibrium (sharp-escape to 2d/η) rather
than lowering it, and the plant's "inertness" is the same statement read at
scale.

This script measures λ_eff along the edited direction in both regimes:

  (A) GD regime (MLP, full batch, theory exact): from the re-run rank-k
      sweep's controller log (figures/rankk_ctrl_s{seed}.csv): the top
      direction's sensed λ̂1 and divisor d give λ_eff = λ̂1/d, plotted
      against the raw λ1 trajectory (which sharp-escapes to ~262) and the
      2/η = 40 boundary.

  (B) Stochastic regime (ResNet-18/CIFAR-100): merges the truelmax plant's
      per-step jsonl (divisor from ARPACK sensing, refreshed every N=20) with
      the outcome ARPACK λ1 (metrics.csv, independent measurement) at the
      outcome steps: λ_eff(t) = λ1(t)/d(t). The plant's damping is real
      (d ≈ 1.9), raw λ1 ≈ 37, so λ_eff ≈ 19–20 ≈ 2/η: the plant PINS the
      preconditioned sharpness at the GD boundary while the raw sharpness
      floats at ~2d/η — the rank-1 analogue of the adaptive-methods
      preconditioned-EoS equilibrium.

Usage:
    python scripts/analysis/analyze_precond_sharpness.py            # both regimes
    python scripts/analysis/analyze_precond_sharpness.py --mlp-only
    python scripts/analysis/analyze_precond_sharpness.py --resnet-only
Outputs: figures/34_precond_sharpness.pdf + figures/precond_sharpness.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import setup  # noqa: E402

LATE_STEP_MLP = 1500
LATE_STEP_R18 = 2000


def ci95(xs):
    from scipy import stats
    xs = np.asarray(xs, float)
    n = len(xs)
    if n < 2:
        return float(xs.mean()), 0.0
    h = float(stats.t.ppf(0.975, n - 1) * xs.std(ddof=1) / np.sqrt(n))
    return float(xs.mean()), h


# ---------------------------------------------------------------- MLP (A) --
def analyze_mlp(seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame | None:
    rows = []
    for s in seeds:
        p = Path(f"figures/rankk_ctrl_s{s}.csv")
        if not p.exists():
            print(f"[mlp] missing {p} — run rankk_sweep_experiment.py --seed {s} first")
            return None
        df = pd.read_csv(p)
        d1 = df[df["i"] == 0].sort_values("step")
        late = d1[d1["step"] >= LATE_STEP_MLP]
        lam_raw = late["lam_hat"].to_numpy()
        div = late["divisor"].clip(lower=1.0).to_numpy()   # clamp arm: d ≥ 1
        lam_eff = lam_raw / div
        rows.append(dict(seed=s,
                         lam1_raw_mean=float(lam_raw.mean()),
                         divisor_mean=float(div.mean()),
                         lam_eff_mean=float(lam_eff.mean()),
                         frac_eff_below_40=float((lam_eff < 40).mean()),
                         frac_raw_below_40=float((lam_raw < 40).mean())))
    out = pd.DataFrame(rows)
    print("\n=== (A) GD regime (MLP rank-k k=1 damping arm, controller-sensed λ̂₁) ===")
    print(f"late window step ≥ {LATE_STEP_MLP}, 2/η = 40")
    print(out.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    m = {c: ci95(out[c].to_numpy()) for c in
         ("lam1_raw_mean", "divisor_mean", "lam_eff_mean")}
    print(f"aggregated: raw λ₁ = {m['lam1_raw_mean'][0]:.2f} ± {m['lam1_raw_mean'][1]:.2f}"
          f" | d = {m['divisor_mean'][0]:.2f} ± {m['divisor_mean'][1]:.2f}"
          f" | λ_eff = λ₁/d = {m['lam_eff_mean'][0]:.2f} ± {m['lam_eff_mean'][1]:.2f}"
          f"  (2/η = 40)")
    return out


# ------------------------------------------------------------- ResNet (B) --
def analyze_resnet(runs=("cifar100_r18_truelmax_s0", "cifar100_r18_truelmax_s1",
                         "cifar100_r18_truelmax_s2", "cifar100_r18_truelmax_s3",
                         "cifar100_r18_truelmax_s4")) -> pd.DataFrame | None:
    rows, traj = [], []
    for r in runs:
        d = Path("runs") / r
        if not (d / "plant_per_step.jsonl").exists():
            print(f"[resnet] missing {d/'plant_per_step.jsonl'} — skip")
            continue
        js = pd.DataFrame([json.loads(l) for l in
                           open(d / "plant_per_step.jsonl")])
        met = pd.read_csv(d / "metrics.csv")
        lr = json.loads((d / "manifest.json").read_text())["lr"]
        # divisor at (or just before) each outcome step: the plant holds the
        # last-sensed value between refreshes — merge_asof semantics
        js_step = js["step"].to_numpy()
        div_at = np.interp(met["step"].to_numpy(), js_step, js["divisor"].to_numpy())
        lam1 = met["lambda_max"].to_numpy()
        late = met["step"].to_numpy() >= LATE_STEP_R18
        lam_eff = lam1 / div_at
        traj.append(pd.DataFrame(dict(run=r, step=met["step"], lam1=lam1,
                                      divisor=div_at, lam_eff=lam_eff)))
        rows.append(dict(run=r,
                         lam1_mean=float(lam1[late].mean()),
                         divisor_mean=float(div_at[late].mean()),
                         lam_eff_mean=float(lam_eff[late].mean()),
                         lam_eff_over_2on_eta=float(lam_eff[late].mean() / (2 / lr)),
                         lam1_over_2on_eta=float(lam1[late].mean() / (2 / lr))))
    if not rows:
        return None
    out = pd.DataFrame(rows)
    print("\n=== (B) stochastic regime (ResNet-18 plant, λ_eff = λ₁(outcome ARPACK)/d(plant)) ===")
    print(f"late window step ≥ {LATE_STEP_R18}, 2/η = 20, HB 2(1+μ)/η = 38")
    print(out.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    m = {c: ci95(out[c].to_numpy()) for c in ("lam1_mean", "divisor_mean", "lam_eff_mean")}
    print(f"aggregated: raw λ₁ = {m['lam1_mean'][0]:.2f} ± {m['lam1_mean'][1]:.2f}"
          f" | d = {m['divisor_mean'][0]:.2f} ± {m['divisor_mean'][1]:.2f}"
          f" | λ_eff = {m['lam_eff_mean'][0]:.2f} ± {m['lam_eff_mean'][1]:.2f}"
          f"  vs 2/η = 20")
    return out


def main():
    setup(scale=1.9093 * 8.5 / 8.69)   # R31: 1/s, s=0.5238 (0.95\linewidth / 718.2pt); ×8.5/8.69 R38 tag-size unification
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--resnet-only", action="store_true")
    args = ap.parse_args()

    mlp_df = resnet_df = None
    if not args.resnet_only:
        mlp_df = analyze_mlp()
    if not args.mlp_only:
        resnet_df = analyze_resnet()

    # ---- figure: (a) MLP raw vs λ_eff; (b) ResNet plant λ₁ vs λ_eff -------
    # (user round, 2026-09-24 fig-review: same treatment as fig 33 —
    # ① the legend moves OUT of the panels onto a single shared row below
    #   (two data series only: raw λ₁ | λ_eff = λ₁/d; unified labels — the
    #   per-regime sensing details live in the caption);
    # ② data strokes thicken 1.6→2.6pt (1.6pt read as hairlines at the
    #   0.95\linewidth include);
    # ③ the threshold rules leave the legend and take right-end direct
    #   tags instead — (a) "2/η=40" above its dashed rule, (b) "2/η=20"
    #   above-left of its rule (right end is occupied by the curves) and
    #   "HB 2(1+μ)/η=38" above its dotted rule at the right end, where the
    #   rendered curves are far below.)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    fs_tag = 0.88 * 15.27
    if mlp_df is not None:
        ax = axes[0]
        trs = []
        for s in sorted(mlp_df["seed"].unique()):
            df = pd.read_csv(f"figures/rankk_ctrl_s{s}.csv")
            d1 = df[df["i"] == 0].sort_values("step")
            trs.append(dict(step=d1["step"].to_numpy(),
                            raw=d1["lam_hat"].to_numpy(),
                            eff=(d1["lam_hat"] / d1["divisor"].clip(lower=1.0)).to_numpy()))
        steps = trs[0]["step"]
        raw_m = np.mean([t["raw"] for t in trs], axis=0)
        eff_m = np.mean([t["eff"] for t in trs], axis=0)
        # R39 grayscale-redundancy lane (fixplan_r15 §6): the two mean
        # curves carry linestyle + marker so they stay separable in
        # grayscale print; markersize rides the setup() rcParam tier
        # (4 pt final). Threshold rules below stay marker-free.
        ax.plot(steps, raw_m, lw=2.6, ls="-", marker="o",
                markevery=max(1, len(steps) // 8), label=r"raw $\lambda_1$")
        ax.plot(steps, eff_m, lw=2.6, ls="--", marker="s",
                markevery=max(1, len(steps) // 8),
                label=r"$\lambda_{\rm eff}=\lambda_1/d$")
        ax.axhline(40, color="k", ls="--", lw=1.3)
        trans_a = mtransforms.blended_transform_factory(ax.transAxes,
                                                        ax.transData)
        ax.text(0.98, 40 * 1.05, r"$2/\eta=40$", transform=trans_a,
                ha="right", va="bottom", fontsize=fs_tag, color="k")
        ax.set_yscale("log")
        ax.set_ylim(1, 600)
        ax.set_xlabel("step"); ax.set_ylabel("sharpness (log)")
        ax.set_title(r"\textbf{(a)} GD regime: damping arm (MLP, n=5)", color="#1A1D21")
        # legend INSIDE panel (a), lower right (user round 2: the external
        # bottom band read as either too close to the xlabels or detached;
        # the lower-right corner of the log panel is entirely ink-free —
        # both curves live at λ≥40, the tag sits above the rule — so the
        # two-entry legend nests there without touching anything)
        ax.legend(loc="lower right", frameon=False, fontsize=15.27,
                  handlelength=1.6, handletextpad=0.5, labelspacing=0.35,
                  borderaxespad=0.4)
    if resnet_df is not None:
        ax = axes[1]
        runs = resnet_df["run"].tolist()
        l1s, les, dvs = [], [], []
        for r in runs:
            d = Path("runs") / r
            js = pd.DataFrame([json.loads(l) for l in open(d / "plant_per_step.jsonl")])
            met = pd.read_csv(d / "metrics.csv")
            div_at = np.interp(met["step"], js["step"], js["divisor"])
            m = met["step"] >= LATE_STEP_R18
            l1s.append(met["lambda_max"].to_numpy()[m])
            les.append((met["lambda_max"].to_numpy() / div_at)[m])
            dvs.append(div_at[m])
        steps = met["step"].to_numpy()[m]
        ax.plot(steps, np.mean(l1s, axis=0), lw=2.6, ls="-", marker="o",
                markevery=max(1, len(steps) // 8), label=r"raw $\lambda_1$")
        ax.plot(steps, np.mean(les, axis=0), lw=2.6, ls="--", marker="s",
                markevery=max(1, len(steps) // 8),
                label=r"$\lambda_{\rm eff}=\lambda_1/d$")
        ax.axhline(20, color="k", ls="--", lw=1.3)
        # (user round 3: the HB rule was gray and its tag floated ~2 data
        # units above it; both fixed — rule black dotted (linestyle already
        # distinguishes it from the dashed 2/η rule), tag tucked to 1.5%
        # of the axis range above the line.)
        ax.axhline(38, color="k", ls=":", lw=1.3)
        trans_b = mtransforms.blended_transform_factory(ax.transAxes,
                                                        ax.transData)
        # direct tags ON the rules (legend entries retired): "2/η=20" sits
        # above-left of its rule — the right end is owned by the curves —
        # while "HB …=38" tags its dotted rule at the right end, where both
        # curves have already descended below 22.
        ax.text(0.02, 20 * 1.06, r"$2/\eta=20$", transform=trans_b,
                ha="left", va="bottom", fontsize=fs_tag, color="k")
        # R39 terminology (fixplan_r15 §7.1): "HB" -> "HB reference" — the
        # rule is a fixed reference, not a theoretical constraint.
        ax.text(0.98, 38 * 1.015, r"HB reference $2(1{+}\mu)/\eta=38$",
                transform=trans_b,
                ha="right", va="bottom", fontsize=fs_tag, color="k")
        ax.set_xlabel("step (late window)"); ax.set_ylabel("sharpness")
        ax.set_title(rf"\textbf{{(b)}} stochastic regime: ARPACK plant (n={len(runs)})", color="#1A1D21")

    fig.tight_layout()
    # R39 atomic regeneration (another session may be compiling the paper):
    # save to <name>.tmp.pdf first, mv over the live PDF only after success.
    out = Path("figures/34_precond_sharpness.pdf")
    tmp = out.with_suffix(".tmp.pdf")
    fig.savefig(tmp, bbox_inches="tight")
    tmp.replace(out)
    csv_rows = []
    if mlp_df is not None:
        csv_rows.append(mlp_df.assign(regime="mlp_gd"))
    if resnet_df is not None:
        csv_rows.append(resnet_df.assign(regime="resnet_plant"))
    if csv_rows:
        pd.concat(csv_rows).to_csv("figures/precond_sharpness.csv", index=False)
    print(f"\n=> {out} + figures/precond_sharpness.csv")


if __name__ == "__main__":
    main()
