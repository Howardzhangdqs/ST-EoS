"""Standard long-training recipe analysis — external validity (review rounds 2+).

Recipes (--recipe):
  100ep  39,100 steps (100 epochs), cosine lr 0.1->0, pad/crop+flip augment,
         3 arms (sgd / accurate rank-1 damping rankkx_d1 / sam); originally
         n=3 seeds.  Steady window epochs 30-70.
  200ep  78,200 steps (200 epochs), same protocol, n=5 seeds (the external-
         review upgrade: full standard budget).  Steady window epochs 30-120
         (cosine lr 0.088->0.023, 19 exact-ARPACK points per seed).

Both recipes measure the reference lambda_max (exact ARPACK, tol 1e-6, k=3,
deterministic 512-sample eval batch) every 5 epochs.  Because the cosine
schedule anneals eta, the operative boundary 2/eta(t) moves; this script
reports lambda_max(t) against 2/eta(t) and 2(1+mu)/eta(t), the arm ordering
in the steady window, late-window values, final accuracies, and — new for
the paired-seed statistics the reviewer asked for — per-seed PAIRED
differences vs SGD (steady-window lambda and final accuracy) with 90% CIs
and a TOST equivalence verdict at the prespecified margin
m = 0.25 * |SGD - SAM| (same rule as analyze_tost_equivalence.py: every arm
runs under deterministic CUDA on the same data-stream seed, so seed s of any
arm saw the identical batch sequence as seed s of SGD).

Outputs: printed summary + figures/38_long_recipe.pdf (100ep) or
figures/39_long_recipe_200ep.pdf (200ep) + the matching summary CSV.
"""
from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steos.analysis.figure_style import setup, C, THEORY, ST

EPOCH = 391  # 50000/128 ~ 390.6

RECIPES = {
    "100ep": dict(
        steps=39100,
        arms=[("SGD", "cifar100_r18_sgd_100ep_s"),
              ("rank-1 damp (accurate)", "cifar100_r18_rankkx_d1_100ep_s"),
              ("SAM", "cifar100_r18_sam_100ep_s")],
        steady_epochs=(30, 70),
        late_epochs=(90, 100),
        fig="figures/38_long_recipe.pdf",
        csv="figures/long_recipe_summary.csv",
    ),
    "200ep": dict(
        steps=78200,
        arms=[("SGD", "cifar100_r18_sgd_200ep_s"),
              ("rank-1 damp (accurate)", "cifar100_r18_rankkx_d1_200ep_s"),
              ("SAM", "cifar100_r18_sam_200ep_s")],
        steady_epochs=(30, 120),
        late_epochs=(180, 200),
        fig="figures/39_long_recipe_200ep.pdf",
        csv="figures/long_recipe_200ep_summary.csv",
    ),
}


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()) if len(xs) else float("nan"), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def paired_tost(deltas: np.ndarray, margin: float) -> dict:
    """Paired mean difference with 90% CI + TOST verdict (prespecified rule)."""
    n = len(deltas)
    mean = float(deltas.mean())
    if n < 2:
        return dict(delta=mean, dlo=mean, dhi=mean, tost_margin=margin,
                    tost_equivalent=False, tost_verdict="n<2 (no test)")
    h = float(stats.t.ppf(0.95, n - 1) * deltas.std(ddof=1) / np.sqrt(n))
    lo, hi = mean - h, mean + h
    equiv = bool(lo > -margin and hi < margin)
    verdict = ("EQUIVALENT" if equiv else "NOT established (CI exits margin)")
    return dict(delta=mean, dlo=lo, dhi=hi, tost_margin=margin,
                tost_equivalent=equiv, tost_verdict=verdict)


def main():
    setup(scale=1.3028 * 8.5 / 8.08 * 8.5 / 8.71)  # R30: source canvas ~1.32x final size (report()-measured); ×8.5/8.08 R38 tag-size unification (2-pass bbox-tight trim)
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="100ep")
    args = ap.parse_args()

    R = RECIPES[args.recipe]
    STEPS = R["steps"]
    STEADY = (R["steady_epochs"][0] * EPOCH, R["steady_epochs"][1] * EPOCH)
    LATE = (R["late_epochs"][0] * EPOCH, R["late_epochs"][1] * EPOCH)

    # collect complete runs per arm: per-seed steady/late lambda, final acc
    per_arm = {}
    for name, pfx in R["arms"]:
        runs = []
        for d in sorted(Path("runs").glob(pfx + "*")):
            s = d.name.rsplit("_s", 1)[-1]
            if s.isdigit() and (d / "metrics.csv").exists():
                df = pd.read_csv(d / "metrics.csv")
                if len(df) and df["step"].iloc[-1] >= STEPS - EPOCH:
                    runs.append((int(s), df))
        if not runs:
            print(f"[skip] {name}: no complete runs yet")
            continue
        per_arm[name] = dict(
            seeds=[s for s, _ in runs],
            lam_st={s: float(df[(df["step"] >= STEADY[0]) & (df["step"] <= STEADY[1])]
                            ["lambda_max"].mean()) for s, df in runs},
            lam_lt={s: float(df[(df["step"] >= LATE[0]) & (df["step"] <= LATE[1])]
                            ["lambda_max"].mean()) for s, df in runs},
            acc={s: float(df["eval_acc"].iloc[-1]) for s, df in runs},
            fired=[float(df["fired_frac"].iloc[-1]) for _, df in runs
                   if "fired_frac" in df and df["fired_frac"].notna().any()],
            runs=runs)
    if not per_arm:
        raise SystemExit(f"no complete long-recipe runs (queue for {args.recipe})")

    # prespecified TOST margin from the SGD/SAM contrast (same rule as the
    # at-scale protocol: m = 0.25 |SGD - SAM| on the steady window)
    margin = float("nan")
    if "SGD" in per_arm and "SAM" in per_arm:
        common = sorted(set(per_arm["SGD"]["seeds"]) & set(per_arm["SAM"]["seeds"]))
        if len(common) >= 2:
            margin = 0.25 * abs(np.mean([per_arm["SGD"]["lam_st"][s] for s in common]) -
                                np.mean([per_arm["SAM"]["lam_st"][s] for s in common]))

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    # R30 style: semantic palette — SGD neutral slate, rank-1 damping blue,
    # SAM teal; annealing boundaries in neutral theory gray.  Stroke
    # hierarchy from ST; legends frameless; fontsize governed by
    # setup(scale) (tick/legend 7.0 final, label 7.5 final).
    colors = {"SGD": C["neutral"], "rank-1 damp (accurate)": C["damping"],
              "SAM": C["sam"]}
    # R39 grayscale-redundancy lane (fixplan_r15 §6): distinct linestyle +
    # marker per arm (SGD '-'/o, rank-1 '--'/s, SAM '-.'/^) so the curves
    # stay separable in grayscale print. Markers ride the bold mean
    # trajectories only (panel (a) curvature checkpoints, markevery=4 for
    # the ~41-point 5-epoch grid); panel (b) inherits the linestyles.
    styles = {"SGD": ("-", "o"), "rank-1 damp (accurate)": ("--", "s"),
              "SAM": ("-.", "^")}
    summary = []
    for name, pfx in R["arms"]:
        if name not in per_arm:
            continue
        A = per_arm[name]
        m1, h1 = ci95(list(A["lam_st"].values()))
        m2, h2 = ci95(list(A["lam_lt"].values()))
        ma, ha = ci95(list(A["acc"].values()))
        row = dict(arm=name, n=len(A["seeds"]),
                   lam_steady=m1, lam_steady_ci=h1,
                   lam_late=m2, lam_late_ci=h2, acc=ma, acc_ci=ha,
                   fired=float(np.mean(A["fired"])) if A["fired"] else float("nan"))
        # paired vs SGD (steady lambda + final accuracy), TOST on lambda
        if "SGD" in per_arm:
            common = sorted(set(A["seeds"]) & set(per_arm["SGD"]["seeds"]))
            if len(common) >= 1:
                dl = np.array([A["lam_st"][s] - per_arm["SGD"]["lam_st"][s]
                               for s in common])
                da = np.array([A["acc"][s] - per_arm["SGD"]["acc"][s]
                               for s in common])
                t = paired_tost(dl, margin) if len(common) >= 2 else \
                    dict(delta=float(dl.mean()), dlo=np.nan, dhi=np.nan,
                         tost_margin=margin, tost_equivalent=False,
                         tost_verdict="n<2 (no test)")
                row.update(d_lam_steady=t["delta"], d_lam_lo=t["dlo"],
                           d_lam_hi=t["dhi"], d_acc=float(da.mean()),
                           d_acc_ci=(float(stats.t.ppf(0.975, len(da) - 1)
                                           * da.std(ddof=1) / np.sqrt(len(da)))
                                     if len(da) >= 2 else np.nan),
                           tost_verdict=t["tost_verdict"])
        summary.append(row)
        tost_txt = (f"  Δλ={row['d_lam_steady']:+.2f} [{row['d_lam_lo']:+.2f},"
                    f"{row['d_lam_hi']:+.2f}] TOST:{row['tost_verdict']}"
                    if "d_lam_steady" in row else "")
        print(f"{name:24s} n={row['n']}  λ(e{R['steady_epochs'][0]}-"
              f"{R['steady_epochs'][1]}) = {m1:5.2f}±{h1:4.2f}  "
              f"λ(e{R['late_epochs'][0]}-"
              f"{R['late_epochs'][1]}) = {m2:5.2f}±{h2:4.2f}  "
              f"acc = {ma:.3f}±{ha:.3f}{tost_txt}")
        if "d_acc" in row and not np.isnan(row.get("d_acc", np.nan)):
            print(f"{'':24s} paired Δacc(SGD) = {row['d_acc']:+.4f}"
                  f" ± {row.get('d_acc_ci', float('nan')):.4f}")

        # panel (a): lambda trajectory vs annealing boundaries (seed-mean)
        runs = A["runs"]
        df0 = runs[0][1]
        steps = df0["step"].to_numpy()
        mean_traj = np.mean([df.set_index("step").reindex(steps)["lambda_max"].to_numpy()
                             for _, df in runs], axis=0)
        lr_t = 0.5 * 0.1 * (1 + np.cos(np.pi * steps / (STEPS - 1)))
        c = colors[name]
        ls_a, mkr = styles[name]
        axes[0].plot(steps / EPOCH, mean_traj, color=c, ls=ls_a, marker=mkr,
                     ms=ST["marker"], markevery=4, lw=ST["primary"],
                     label=name)
        if name == "SGD":
            axes[0].plot(steps / EPOCH, 2 / lr_t, color=THEORY["dark"], ls="--",
                         lw=ST["theory"], label=r"$2/\eta(t)$")
            # R39 terminology (fixplan_r15 §7.1): "HB" -> "HB ref." (fixed
            # reference, not a constraint).
            axes[0].plot(steps / EPOCH, 2 * 1.9 / lr_t, color=THEORY["light"],
                         ls=":", lw=ST["theory"],
                         label=r"HB ref. $2(1{+}\mu)/\eta(t)$")
        # (b) accuracy: the three arms sit within ~0.01 of each other — solid
        # lines fully occlude; arms carry their R39 linestyles so all three
        # stay visible (display-layer only, color remains the arm channel);
        # marker-free (the curves coincide, markers would clump).
        axes[1].plot(steps / EPOCH, np.mean(
            [df.set_index("step").reindex(steps)["eval_acc"].to_numpy()
             for _, df in runs], axis=0), color=c, lw=ST["primary"], ls=ls_a,
            label=name)
        axes[2].errorbar(m1, ma, xerr=h1, yerr=ha, fmt=mkr, color=c,
                         capsize=ST["capsize"], label=name)

    if np.isfinite(margin):
        print(f"\nprespecified TOST margin m = 0.25·|SGD−SAM| steady = {margin:.3f}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(r"reference $\lambda_{\mathrm{ref}}$")
    axes[0].set_title(r"\textbf{(a)} sharpness vs annealed boundaries", color="#1A1D21",)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("eval acc (2k test)")
    axes[1].set_title(r"\textbf{(b)} accuracy", color="#1A1D21",)
    axes[2].set_xlabel(rf"$\lambda_{{\mathrm{{ref}}}}$ epochs {R['steady_epochs'][0]}–"
                       f"{R['steady_epochs'][1]}"); axes[2].set_ylabel("final acc")
    axes[2].set_title(r"\textbf{(c)} sharpness–accuracy", color="#1A1D21",)
    fig.tight_layout()
    # R38 (user): single shared legend at the figure BOTTOM, one horizontal
    # row — replaces the three per-panel legends. Added AFTER tight_layout
    # (outside legends don't participate in it; savefig's bbox_inches="tight"
    # re-expands the canvas to include it). Handles deduped across panels:
    # arm labels repeat in (a)/(b)/(c); the two annealed-boundary lines
    # exist only in (a).
    seen, handles, labels = set(), [], []
    for ax in (axes[0], axes[2], axes[1]):   # (a) solid arm lines + theory first
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    # display order: the three arms together, then the annealed-boundary
    # rules (sample style = (a)'s solid arm lines)
    order = ["SGD", "rank-1 damp (accurate)", "SAM",
             r"$2/\eta(t)$", r"HB ref. $2(1{+}\mu)/\eta(t)$"]
    by_label = dict(zip(labels, handles))
    pairs = [(by_label[k], k) for k in order if k in by_label]
    pairs += [(h, l) for h, l in zip(handles, labels) if l not in order]
    handles = [h for h, _ in pairs]
    labels = [l for _, l in pairs]
    # (cross-figure key spacing: gap to the panels ~6 pt at 0.95\linewidth
    # in the paper — see R38 audit.)
    fig.legend(handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 0.0795), ncol=len(labels),
                frameon=False, columnspacing=0.5, handlelength=1.4,
                handletextpad=0.4)
    # R39 atomic regeneration (another session may be compiling the paper):
    # save to <name>.tmp.pdf first, mv over the live PDF only after success.
    outp = Path(R["fig"])
    tmp = outp.with_suffix(".tmp.pdf")
    fig.savefig(tmp, bbox_inches="tight")
    tmp.replace(outp)
    pd.DataFrame(summary).to_csv(R["csv"], index=False)
    print(f"=> {outp} + {R['csv']}")


if __name__ == "__main__":
    main()
