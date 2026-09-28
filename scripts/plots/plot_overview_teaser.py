"""Render the paper overview / teaser figure (PDF + PNG).

Generates ``figures/00_overview_teaser.pdf`` and ``.png`` — a three-panel
summary of the ST-EoS project:

  (a) control-loop schematic: training loop with observer (monitor) vs.
      local actuator branches around the EoS equilibrium 2/eta
  (b) Prediction panel: cheap 1-HVP proxy tracks lambda_max (r ~= 0.98)
  (c) Control panel: one-sidedly non-pushable cap — local actuators pin
      or escape the WRONG way; only global SAM flat-escapes

All curves are synthetic/illustrative (no data files are read).

Regenerate with (from repo root)::

    uv run python scripts/plots/plot_overview_teaser.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rng = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#666666"

# ------------------------------------------------------------------ layout
fig = plt.figure(figsize=(13.2, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25],
                      left=0.045, right=0.99, top=0.845, bottom=0.13,
                      wspace=0.28)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

fig.suptitle(
    "Sharpness proxies are predictors, not local actuators:\n"
    "the Edge of Stability supports monitoring, not local curvature control",
    fontsize=11.5, fontweight="bold", y=0.985,
)

# ============================================================== panel (a)
axA.set_title("(a) GD training loop: observe vs. actuate",
              fontsize=10, fontweight="bold", pad=8)
axA.set_xlim(0, 10)
axA.set_ylim(0, 7.4)
axA.set_axis_off()


def box(ax, cx, cy, w, h, edge, face, lw=1.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.06",
                                edgecolor=edge, facecolor=face, lw=lw,
                                mutation_aspect=0.72))


def arrow(ax, x0, y0, x1, y1, color=GRAY, style="-|>", ls="-", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=12, lw=lw, color=color,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


# main loop boxes
box(axA, 5.0, 6.55, 3.6, 0.78, "#333333", "#f5f5f5")
axA.text(5.0, 6.55, r"Data $\to$ Loss $L(\theta)$", ha="center", va="center",
         fontsize=8.5, fontweight="bold")

box(axA, 5.0, 5.35, 2.6, 0.66, "#333333", "#f5f5f5")
axA.text(5.0, 5.35, r"Gradient $g$", ha="center", va="center",
         fontsize=8.5, fontweight="bold")

box(axA, 5.0, 4.25, 4.0, 0.82, "#111111", "#eeeeee", lw=2.0)
axA.text(5.0, 4.45, "Optimizer step", ha="center", va="center",
         fontsize=8.5, fontweight="bold")
axA.text(5.0, 4.08, r"$\theta \leftarrow \theta - \eta \nabla$",
         ha="center", va="center", fontsize=8)

arrow(axA, 5.0, 6.13, 5.0, 5.72)
arrow(axA, 5.0, 4.99, 5.0, 4.70)

# training-loop return arrow (right side)
arrow(axA, 7.05, 4.25, 6.85, 6.55, rad=-0.45, lw=1.2)
axA.text(8.75, 5.45, "training loop", ha="center", va="center",
         fontsize=7, color=GRAY, rotation=90)

# observer branch (left, blue)
box(axA, 2.2, 3.15, 4.1, 1.15, BLUE, "#e3f0fb")
axA.text(2.2, 3.48, "Monitor  (observer)", ha="center", va="center",
         fontsize=8.5, fontweight="bold", color=BLUE)
axA.text(2.2, 2.92, r"HVP probe:  $\hat{\lambda}_g = g^{\top} H g\, /\, g^{\top} g$",
         ha="center", va="center", fontsize=7.6)
axA.text(2.2, 2.30, r"proxy ladder:  Tier-0 $\|W\|,\gamma,\|g\|$",
         ha="center", va="center", fontsize=7.2, color=GRAY)
axA.text(2.2, 1.96, r"$\to$ 1-HVP $\to$ warm-start power-iter $\to$ ARPACK",
         ha="center", va="center", fontsize=7.2, color=GRAY)
arrow(axA, 3.68, 5.28, 2.55, 3.78, color=BLUE, ls="--", rad=0.25)
axA.text(2.2, 1.42, "\u2713  predicts", ha="center", va="center",
         fontsize=10, fontweight="bold", color=BLUE)

# actuator branch (right, red)
box(axA, 7.8, 3.15, 4.1, 1.15, RED, "#fbe7e5")
axA.text(7.8, 3.48, "Actuator  (local controller)", ha="center", va="center",
         fontsize=8.5, fontweight="bold", color=RED)
axA.text(7.8, 2.92, r"gradient edit:  $\nabla \leftarrow \nabla / d$,"
         "\n"
         r"$d = \hat{\lambda} / (2/\eta)$",
         ha="center", va="center", fontsize=7.6)
axA.text(7.8, 2.28, r"rank-1 / rank-$k$ plant actions  (C1–C4, X-SAM)",
         ha="center", va="center", fontsize=7.2, color=GRAY)
arrow(axA, 6.32, 5.28, 7.45, 3.78, color=RED, ls="--", rad=-0.25)
axA.text(7.8, 1.42, "\u2717  cannot steer", ha="center", va="center",
         fontsize=10, fontweight="bold", color=RED)

# EoS equilibrium dashed line
axA.plot([0.3, 9.7], [0.72, 0.72], ls="--", lw=1.3, color="#444444")
axA.text(5.0, 0.30,
         r"EoS equilibrium:  $\lambda_{\max} \to 2/\eta$"
         "   (self-stabilization cap)",
         ha="center", va="center", fontsize=7.6, color="#333333")

# ============================================================== panel (b)
axB.set_title("(b) Prediction \u2713 — cheap proxies track "
              r"$\lambda_{\max}$", fontsize=10, fontweight="bold", pad=8)
k = np.linspace(0, 400, 400)
lam = 2.0 * (1 - np.exp(-k / 60))
lam += 0.015 * np.sin(k / 2.2)                      # tiny period-2 wiggle
noise = rng.normal(0, 0.05, k.size)
sm = np.convolve(noise, np.ones(9) / 9, mode="same")
proxy = lam + sm * 0.7

axB.axhline(2.0, color="black", ls="--", lw=1.0)
axB.text(396, 2.04, r"$2/\eta$", ha="right", va="bottom", fontsize=8.5)
axB.plot(k, lam, color="#08519c", lw=1.8,
         label=r"$\lambda_{\max}$ (ARPACK ground truth)")
axB.plot(k, proxy, color="#f28e2b", lw=1.1, ls=":",
         markevery=14, marker="o", ms=2.4,
         label=r"$\hat{\lambda}_g$ (1 HVP, raw gradient)")
axB.set_xlim(0, 400)
axB.set_ylim(0, 2.35)
axB.set_xlabel("training step $k$", fontsize=9)
axB.set_ylabel(r"$\lambda_{\max}$", fontsize=9)
axB.tick_params(labelsize=8)
axB.legend(fontsize=7, loc="upper left", framealpha=0.9,
           handlelength=1.8, borderpad=0.4)
axB.text(0.97, 0.05,
         "EoS-phase detrended corr:  r = +0.98\n"
         "momentum-smoothed probes fail; monitor raw $g$",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=7,
         bbox=dict(boxstyle="round,pad=0.35", fc="#e3f0fb", ec=BLUE, lw=0.7))

# ============================================================== panel (c)
axC.set_title("(c) Control \u2717 — one-sidedly non-pushable cap",
              fontsize=10, fontweight="bold", pad=8)
rise = 0.6 + 0.4 * (1 - np.exp(-k / 45))
pinned = np.where(k < 60, rise, 1.0 + 0.02 * np.sin(k / 2.2))

gd = pinned
caps = np.where(k < 60, rise, 1.0)
lagged = np.where(k < 70, rise, 1.0 + 0.012 * np.sin(k / 2.6))
sharp = np.where(k <= 200, pinned, np.exp((k - 200) / 41.0))
sharp = np.clip(sharp, None, 130.0)
ortho = np.where(k < 75, rise, 1.0)
sam = 0.6 + 0.3 * (1 - np.exp(-k / 45))
sam = np.where(k > 110, 0.9 - 0.35 * (1 - np.exp(-(k - 110) / 70)), sam)

axC.axhline(1.0, color="black", ls="--", lw=1.0)
axC.text(396, 1.06, r"$2/\eta$", ha="right", va="bottom", fontsize=8.5)
axC.plot(k, gd, color="#555555", lw=1.4, label="GD (EoS)")
axC.plot(k, caps, color="#999999", lw=1.2, ls="-.",
         label=r"W-cap / $\gamma$-cap (C1/C2)")
axC.plot(k, lagged, color="#2ca02c", lw=1.2, ls=":",
         label=r"rank-1 plant, lagged $\hat{v}$ (C3-i)")
axC.plot(k, sharp, color=RED, lw=2.0,
         label=r"rank-1 plant, exact $v_{\mathrm{top}}$ (C3-ii)")
axC.plot(k, ortho, color="#9467bd", lw=1.2, ls="--",
         label="orth.-complement damping (C4)")
axC.plot(k, sam, color="#17becf", lw=2.0,
         label="SAM (global action) \u2713 flat-escapes")

axC.annotate("sharp-escape\n"
             r"($\lambda_{\max} \approx 263$ vs cap $2/\eta$)"
             "\nmoves the WRONG way",
             xy=(392, 122), xytext=(210, 30), fontsize=6.8, color=RED,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
axC.text(120, 1.30,
         "self-stabilization cap:\nperiod-2 oscillation re-supplies sharpness",
         ha="center", va="bottom", fontsize=6.6, color="#555555")
axC.text(0.03, 0.03,
         "Pin(a), pinned-despite-action:\nsharpness re-distributes over the "
         r"rank-$(n{-}1)$ complement",
         transform=axC.transAxes, ha="left", va="bottom", fontsize=6.6,
         bbox=dict(boxstyle="round,pad=0.35", fc="#fbe7e5", ec=RED, lw=0.7))

axC.set_yscale("log")
axC.set_ylim(0.3, 140)
axC.set_xlim(0, 400)
axC.set_xlabel("training step $k$", fontsize=9)
axC.set_ylabel(r"$\lambda_{\max}\, /\, (2/\eta)$", fontsize=9)
axC.tick_params(labelsize=8)
axC.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.92,
           columnspacing=1.0, handlelength=1.7, borderpad=0.4)
axC.spines["right"].set_visible(True)

# ------------------------------------------------------------------ strip
fig.text(0.5, 0.015,
         r"$\hat{\lambda}_g$ from a single HVP predicts EoS (r $\approx$ 0.98)"
         "    \u2022    "
         r"rank-$k$ local edits never push below $2/\eta$"
         "    \u2022    "
         "SAM escapes; the sharp-escape arm even generalizes best "
         "(\u221218.5% test MSE)",
         ha="center", va="bottom", fontsize=8, color="#222222")

# ------------------------------------------------------------------ save
fig.savefig("figures/00_overview_teaser.pdf", bbox_inches="tight",
            facecolor="white")
fig.savefig("figures/00_overview_teaser.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("saved figures/00_overview_teaser.{pdf,png}")
