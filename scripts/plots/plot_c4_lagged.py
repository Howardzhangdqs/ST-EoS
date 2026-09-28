"""Fig 31 (fig:c4_lagged_controls) — two control experiments on the rank-1 plant.

Two panels side-by-side:

  (a) C4 orthogonal damping sweep: x = d in {1, 1.5, 2, 4, 8}, y = final
      λ_max (5-seed mean ± 95% CI). All d pin at 2/η = 40 — the
      self-stabilization cap is robust to the orthogonal-damping strength.
      A thin lower strip (shared x axis, height ratio ~4:1) shows final loss
      rising 0.11 → 0.59 as d increases, so heavier damping does NOT unblock
      sub-2/η escape; it only hurts optimization.

  (b) Lagged sensing sweep: x = k in {0, 1, 2, 3}, y = final λ_max (5-seed
      mean ± 95% CI), log y-axis. k=0 (no lag, full sharpness visible to the
      rank-1 controller) escapes to λ_max ≈ 262 — far above 2/η. k ≥ 1
      (controller sees a stale sharpness estimate) collapses onto 2/η = 40
      exactly: the lag breaks the alignment that lets the rank-1 controller
      "see" and chase the top eigen-direction, so it can no longer act and
      the GD self-stabilization cap re-imposes itself.

R30 restyle (aesthetic_diagnosis_r29.md §4B p5, lane D4):
  * (a) the old twin-y axis (blue λraw left + red loss right) is split into
    a main λraw panel + a thin lower loss strip sharing the x axis — one
    quantity per axis, no second colored axis;
  * (b) the giant diamond/triangle markers and the boxed legend are replaced
    by restrained ST-size markers and line-end direct labels. Both sensing
    arms belong to the damping family (same hue): the Δθ arm is a solid
    line with filled circles at k ≥ 1 and an open circle at the no-lag
    k = 0 escape; the û arm is a dashed line with open triangles. The 2/η
    threshold is a neutral theory-gray rule in both panels.

Reads:  runs/c4/analysis.json
        runs/lagged/analysis.json
Writes: figures/31_c4_lagged_controls.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.analysis.figure_style import (  # noqa: E402
    C,
    ST,
    THEORY,
    panel_label,
    report,
    savefig,
    setup,
)

# Source canvas is included at 0.95\linewidth and shrunk accordingly; the
# scale factor keeps the FINAL on-paper fonts at the R30 floors (see the
# report() output at the end of the run).
setup(scale=1.2017)

RUNS = Path("runs")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

C4_JSON = RUNS / "c4" / "analysis.json"
LAG_JSON = RUNS / "lagged" / "analysis.json"

# direct-label / threshold-tag size (pt on the source canvas)
FS_TAG = ST["note"]


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text())


def main():
    c4 = load(C4_JSON)
    lag = load(LAG_JSON)
    two_over_eta = float(c4.get("two_over_eta", lag.get("two_over_eta", 40.0)))

    # ---------------- C4 (panel a) ----------------
    c4_tab = c4["table"]
    d_vals = np.array([row["d"] for row in c4_tab], dtype=float)
    lam_mean = np.array([row["lambda_max_final_mean"] for row in c4_tab], dtype=float)
    lam_ci = np.array([row["lambda_max_final_ci"] for row in c4_tab], dtype=float)
    loss_mean = np.array([row["loss_final_mean"] for row in c4_tab], dtype=float)
    loss_ci = np.array([row["loss_final_ci"] for row in c4_tab], dtype=float)
    n_seeds_c4 = int(c4_tab[0]["n_seeds"])

    # ---------------- Lagged (panel b) ----------------
    # Δθ-based arm (existing): ALL k values
    lag_tab = lag["table"]
    dt_rows = [r for r in lag_tab if r.get("sensing", "dtheta") == "dtheta"]
    eig_rows = [r for r in lag_tab if r.get("sensing") == "eigvec"]

    k_vals = np.array([row["k"] for row in dt_rows], dtype=float)
    lam_mean_k = np.array([row["lambda_max_final_mean"] for row in dt_rows], dtype=float)
    lam_ci_k = np.array([row["lambda_max_final_ci"] for row in dt_rows], dtype=float)
    n_seeds_lag = int(dt_rows[0]["n_seeds"]) if dt_rows else 5

    # û-based arm (accurate eigenvector, lagged): k ∈ {1, 2, 3}
    k_vals_eig = np.array([row["k"] for row in eig_rows], dtype=float) if eig_rows else np.array([])
    lam_mean_eig = np.array([row["lambda_max_final_mean"] for row in eig_rows], dtype=float) if eig_rows else np.array([])
    lam_ci_eig = np.array([row["lambda_max_final_ci"] for row in eig_rows], dtype=float) if eig_rows else np.array([])

    # ---------------- figure ----------------
    # Left column: (a) main λraw panel + thin loss strip (shared x, ~4:1).
    # Right column: (b) lagged sensing, spanning both rows.
    fig = plt.figure(figsize=(6.5, 3.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.0], width_ratios=[1.0, 1.12],
                          hspace=0.14, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_b = fig.add_subplot(gs[:, 1])
    fig.subplots_adjust(left=0.094, right=0.988, top=0.895, bottom=0.125)

    blue = C["damping"]  # rank-1 damping family hue (both sensing arms)

    # ---------- Panel (a) main: final λ_max vs d ----------
    ax_a.errorbar(d_vals, lam_mean, yerr=lam_ci,
                  fmt="o-", color=blue, ecolor=blue,
                  elinewidth=ST["errbar"], capsize=ST["capsize"],
                  lw=ST["primary"], markersize=ST["marker"],
                  markerfacecolor=blue, markeredgecolor=blue, zorder=3)
    ax_a.axhline(two_over_eta, color=THEORY["dark"], ls="--",
                 lw=ST["theory"], zorder=2)
    ax_a.set_xscale("log")
    xticks = list(d_vals)
    ax_a.set_xticks(xticks)
    ax_a.set_xticklabels([f"{v:g}" for v in xticks])
    # log axis spanning <1 decade (1–8): matplotlib's log formatter labels the
    # MINOR ticks (3, 6) in scientific notation (3×10⁰, 6×10⁰) mixed with the
    # plain major labels (1, 1.5, 2, 4, 8) — suppress minor tick labels
    ax_a.tick_params(axis="x", which="minor", labelbottom=False)
    # x tick labels live on the loss strip below (shared x)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    ax_a.set_ylabel(r"final $\lambda_{\mathrm{raw}}$")
    # keep linear y; all means in ~31–49 but include CI headroom
    y_lo = min((lam_mean - lam_ci).min(), two_over_eta) - 4.0
    y_hi = (lam_mean + lam_ci).max() + 4.0
    ax_a.set_ylim(y_lo, y_hi)
    # threshold tag: below-left of the rule, in the clear band under the
    # d ≤ 2 cluster (checked against every whisker bottom)
    trans_a = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
    ax_a.text(0.02, two_over_eta - 0.16 * (two_over_eta - y_lo),
              fr"$2/\eta = {two_over_eta:g}$",
              transform=trans_a, ha="left", va="top",
              fontsize=FS_TAG, color=THEORY["dark"])
    panel_label(ax_a, "a", "Orthogonal damping")

    # ---------- Panel (a) strip: final loss vs d (same x) ----------
    ax_loss.errorbar(d_vals, loss_mean, yerr=loss_ci,
                     fmt="o-", color=C["neutral"], ecolor=C["neutral"],
                     elinewidth=ST["errbar"], capsize=ST["capsize"],
                     lw=ST["secondary"], markersize=0.85 * ST["marker"],
                     markerfacecolor=C["neutral"], markeredgecolor=C["neutral"],
                     zorder=3)
    ax_loss.set_xscale("log")
    ax_loss.set_xticks(xticks)
    ax_loss.set_xticklabels([f"{v:g}" for v in xticks])
    ax_loss.tick_params(axis="x", which="minor", labelbottom=False)
    ax_loss.set_xlabel(r"orthogonal damping  $d$")
    ax_loss.set_ylabel("final loss")
    loss_lo = max(0.0, (loss_mean - loss_ci).min() - 0.05)
    loss_hi = (loss_mean + loss_ci).max() + 0.10
    ax_loss.set_ylim(loss_lo, loss_hi)
    ax_loss.yaxis.set_major_locator(MaxNLocator(3))

    # ---------- Panel (b): lagged sensing ----------
    # guide line through the Δθ arm (log-scale friendly), then markers
    ax_b.plot(k_vals, lam_mean_k, "-", color=blue, lw=ST["secondary"],
              alpha=0.8, zorder=2)
    # k = 0 (no lag, sharp-escape): OPEN circle — the open/filled channel
    # encodes the outcome (escape vs pin) redundantly with position
    is_escape = k_vals == 0
    ax_b.errorbar(k_vals[is_escape], lam_mean_k[is_escape],
                  yerr=lam_ci_k[is_escape],
                  fmt="o", color=blue, ecolor=blue,
                  elinewidth=ST["errbar"], capsize=ST["capsize"],
                  markersize=ST["marker"], markerfacecolor="none",
                  markeredgecolor=blue, zorder=3)
    # k ≥ 1 (lagged, pin at 2/η): filled circles
    ax_b.errorbar(k_vals[~is_escape], lam_mean_k[~is_escape],
                  yerr=lam_ci_k[~is_escape],
                  fmt="o", color=blue, ecolor=blue,
                  elinewidth=ST["errbar"], capsize=ST["capsize"],
                  markersize=ST["marker"], markerfacecolor=blue,
                  markeredgecolor=blue, zorder=3)
    ax_b.axhline(two_over_eta, color=THEORY["dark"], ls="--",
                 lw=ST["theory"], zorder=1.5)
    ax_b.set_xticks(k_vals)
    ax_b.set_xticklabels([f"{int(v)}" for v in k_vals])
    ax_b.set_xlabel(r"sensing lag  $k$  (steps)")
    ax_b.set_ylabel(r"final $\lambda_{\mathrm{raw}}$")
    ax_b.set_yscale("log")
    # generous y-range so the 262 marker sits cleanly above the 40 cluster
    ax_b.set_ylim(20.0, 400.0)
    # room on the right for the two line-end direct labels
    ax_b.set_xlim(-0.28, 4.42)

    # û-based arm (accurate eigenvector, lagged) — dashed + open triangles
    if len(k_vals_eig) > 0:
        x_offset = 0.18  # horizontal offset for clear visual separation
        ax_b.plot(k_vals_eig + x_offset, lam_mean_eig,
                  "--", color=blue, lw=ST["secondary"], alpha=0.8, zorder=2)
        ax_b.errorbar(k_vals_eig + x_offset, lam_mean_eig, yerr=lam_ci_eig,
                      fmt="^", color=blue, ecolor=blue,
                      elinewidth=ST["errbar"], capsize=ST["capsize"],
                      markersize=ST["marker"], markerfacecolor="none",
                      markeredgecolor=blue, zorder=3)
        # line-end direct label (replaces the legend entry)
        ax_b.text(k_vals_eig[-1] + x_offset + 0.10, lam_mean_eig[-1],
                  r"$\hat u(t{-}k)$",
                  ha="left", va="center", fontsize=FS_TAG, color=blue)

    # line-end direct label for the Δθ arm — placed BELOW the 2/η threshold
    # line (R30.1 user feedback): the previous va=center label sat exactly on
    # the black dashed rule and read as one overprinted blob. The k≥1 points
    # pin AT 2/η, so the natural anchor is the rule itself; labeling just
    # below keeps the series identity while leaving the rule unobstructed.
    ax_b.annotate(r"$\Delta\theta_{t-k}$",
                  xy=(k_vals[-1], two_over_eta), xycoords="data",
                  xytext=(10, -8), textcoords="offset points",
                  ha="left", va="top", fontsize=FS_TAG, color=blue)

    # threshold tag at the left end of the rule (clear of the descending arm)
    ax_b.text(-0.20, two_over_eta * 1.03,
              fr"$2/\eta = {two_over_eta:g}$",
              ha="left", va="bottom", fontsize=FS_TAG, color=THEORY["dark"])

    # escape-level tag near the k = 0 point. The escape level is shown as
    # ≈262 (this arm's deterministic mean is 262.2; the canonical n=5
    # rank-k sweep escape value is 261.8±5.4 — the arm mean sits within its
    # CI, so both round to ~262).
    ax_b.text(k_vals[0] + 0.07, 296.0,
              r"$\lambda_{\mathrm{raw}}\approx 262$",
              ha="left", va="bottom", fontsize=FS_TAG, color=THEORY["dark"])

    panel_label(ax_b, "b", "Lagged sensing")

    out = savefig(fig, "31_c4_lagged_controls")
    print(f"[fig] {out}")
    report(out, "31_c4_lagged_controls")

    # ---------------- report numbers (sanity vs analysis.json) ----------------
    print("\n=== Panel (a) C4 — final λ_max & final loss (mean ± CI95) ===")
    print(f"  2/η = {two_over_eta:g}")
    print(f"  {'d':>5s}  {'λ_max':>16s}  {'loss':>16s}")
    for d, m, c, lm, lc in zip(d_vals, lam_mean, lam_ci, loss_mean, loss_ci):
        print(f"  {d:>5g}  {m:>8.3f} ± {c:6.3f}  {lm:>8.4f} ± {lc:6.4f}")

    print("\n=== Panel (b) Lagged — final λ_max (mean ± CI95) ===")
    print(f"  2/η = {two_over_eta:g}")
    print(f"  {'arm':>8s} {'k':>3s}  {'λ_max':>16s}  {'class':>20s}")
    for row in dt_rows:
        m = row["lambda_max_final_mean"]; c = row["lambda_max_final_ci"]
        cls = row.get("classification", "")
        print(f"  {'Δθ':>8s} {int(row['k']):>3d}  {m:>8.3f} ± {c:6.3f}  {cls:>20s}")
    for row in eig_rows:
        m = row["lambda_max_final_mean"]; c = row["lambda_max_final_ci"]
        cls = row.get("classification", "")
        print(f"  {'û':>8s} {int(row['k']):>3d}  {m:>8.3f} ± {c:6.3f}  {cls:>20s}")


if __name__ == "__main__":
    main()
