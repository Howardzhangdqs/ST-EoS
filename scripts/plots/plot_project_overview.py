"""Render the project overview figure (PDF + PNG): single rounded frame.

Generates ``figures/00_project_overview.pdf`` / ``.png`` — one large rounded
rectangle containing the five-stage pipeline of the ST-EoS project:

  1 Training loop (plant)  ->  2 Measure sharpness (proxy ladder)
  ->  3 Prediction (works)  /  4 Local control (fails)  ->  5 Mechanism & verdict

Style follows the project-wide convention (``steos.analysis.figure_style``):
vector PDF, Times text + Computer Modern math via usetex — identical to all
other figures in ``figures/``.

Regenerate with (from repo root)::

    .venv/bin/python scripts/plots/plot_project_overview.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from steos.analysis.figure_style import setup

setup()
# figure_style preamble has amsmath+times; add amssymb for \checkmark
plt.rcParams["text.latex.preamble"] += r"\usepackage{amssymb}"

BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
GRAY = "#555555"

fig = plt.figure(figsize=(12.8, 6.65))
ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.set_axis_off()


def box(x0, y0, w, h, edge, face, lw=1.4):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.25,rounding_size=1.1",
                                edgecolor=edge, facecolor=face, lw=lw))


def arrow(x0, y0, x1, y1, color=GRAY, lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def txt(x, y, s, size=9.0, color="#222222", weight="normal", style="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, fontstyle=style)


# ------------------------------------------------------------- outer frame
box(1.2, 1.2, 97.6, 49.6, "#777777", "#fbfbfb", lw=1.6)
txt(50, 48.0,
    r"\textbf{ST-EoS: sharpness--trajectory co-analysis of the Edge of Stability}",
    size=13, color="#111111")
txt(50, 45.4,
    r"\textit{cheap sharpness proxies predict training, but local curvature "
    r"control cannot steer it}",
    size=9.5, color=GRAY)

# ------------------------------------------------- (1) training loop (plant)
box(3.0, 12.0, 21.5, 28.0, GRAY, "#f5f5f5")
cx = 13.75
txt(cx, 38.2, r"\textbf{1~Training loop (the plant)}", size=10.5)
txt(cx, 35.7, r"data $\to$ loss $L(\theta)$", size=9)
txt(cx, 33.7, r"gradient $\mathbf{g}$", size=9)
txt(cx, 31.6, r"$\theta \leftarrow \theta - \eta \mathbf{g}$", size=9)
txt(cx, 29.6, r"(GD; SGD$+$momentum: $2(1{+}\mu)/\eta$)", size=8, color=GRAY)
# loop-back arrow inside the box
arrow(5.6, 31.4, 5.6, 36.0, color=GRAY, lw=1.2, rad=0.55)
txt(cx, 26.6, r"\textit{setups:}", size=8.5, color=GRAY)
txt(cx, 24.4, r"teacher--student tanh MLP ($n{=}1024$)", size=8.5)
txt(cx, 22.4, r"CIFAR-10 / CIFAR-100", size=8.5)
txt(cx, 20.0, r"MLP $\cdot$ ResNet-18 (BN) $\cdot$", size=8.5)
txt(cx, 18.0, r"ViT-tiny / ConvNeXt-atto (LN)", size=8.5)
txt(cx, 15.0, r"per-step probes + weight snapshots", size=8, color=GRAY)

# ------------------------------------------------ (2) measure sharpness
box(27.5, 12.0, 21.5, 28.0, GRAY, "#f5f5f5")
cx = 38.25
txt(cx, 38.2, r"\textbf{2~Measure sharpness (observer)}", size=10.5)
txt(cx, 35.6, r"\textit{Hessian--vector probes on raw $\mathbf{g}$:}", size=8.5,
    color=GRAY)
txt(cx, 33.2, r"0 HVP -- $\|\mathbf{W}\|$, BN $\gamma$, $\|\mathbf{g}\|$", size=9)
txt(cx, 31.0, r"1 HVP -- $\hat{\lambda}_g = "
              r"\mathbf{g}^{\top} H \mathbf{g}\, /\, \mathbf{g}^{\top}\mathbf{g}$",
    size=9)
txt(cx, 28.8, r"$\approx$ 5 HVP -- warm-started power iteration", size=9)
txt(cx, 26.6, r"exact -- ARPACK $\lambda_{\max}$ (ground truth)", size=9)
txt(cx, 23.2, r"\textit{proxy ladder: cheap $\to$ exact,}", size=8.5, color=GRAY)
txt(cx, 21.2, r"\textit{monitor raw $\mathbf{g}$, not momentum}", size=8.5,
    color=GRAY)
txt(cx, 19.2, r"\textit{smoothed $\Delta\theta$ (false convergence)}", size=8.5,
    color=GRAY)

# ------------------------------------------------------- (3) prediction OK
box(52.5, 26.5, 22.5, 17.5, BLUE, "#eaf2fb")
cx = 63.75
txt(cx, 42.1, r"\textbf{3~Prediction} $\checkmark$", size=10.5, color=BLUE)
txt(cx, 39.4, r"$\hat{\lambda}_g$ tracks $\lambda_{\max}$ through", size=9)
txt(cx, 37.3, r"the EoS phase: detrended $r = +0.98$", size=9)
txt(cx, 34.4, r"EoS equilibrium: $\lambda_{\max} \nearrow 2/\eta$", size=9)
txt(cx, 32.3, r"then locks (progressive sharpening)", size=9)
txt(cx, 29.2, r"\textit{cheap, seed-stable monitoring}", size=8.5, color=BLUE)

# -------------------------------------------------------- (4) control fails
box(52.5, 6.0, 22.5, 17.5, RED, "#fceceb")
cx = 63.75
txt(cx, 21.6, r"\textbf{4~Local control} $\times$", size=10.5, color=RED)
txt(cx, 18.9, r"\textit{proxy-driven gradient edits:}", size=9)
txt(cx, 16.8, r"C1 $W$-cap $\cdot$ C2 $\gamma$-cap $\cdot$", size=8.5)
txt(cx, 14.8, r"C3 rank-1 plant $\cdot$ C4 orth.-complement", size=8.5)
txt(cx, 12.8, r"rank-$k$ sweep $\cdot$ X-SAM", size=8.5)
txt(cx, 10.0, r"\textit{never below $2/\eta$: pinned (Pin(a));}", size=8.5,
    color=RED)
txt(cx, 7.9, r"\textit{exact $v_{\mathrm{top}}$: sharp-escape $\approx 263$}",
    size=8.5, color=RED)

# ------------------------------------------------- (5) mechanism & verdict
box(78.0, 6.0, 19.0, 38.0, "#333333", "#f2f2f2")
cx = 87.5
txt(cx, 41.8, r"\textbf{5~Mechanism \& verdict}", size=10.5)
txt(cx, 38.7, r"the cap is a \textit{self-stabilization}", size=9)
txt(cx, 36.7, r"equilibrium: the period-2 EoS oscillation", size=9)
txt(cx, 34.7, r"re-supplies sharpness at $2/\eta$", size=9)
txt(cx, 31.4, r"$\Rightarrow$ \textbf{one-sidedly non-pushable:}", size=9)
txt(cx, 29.2, r"rank-$k$ local edits cannot cross it;", size=9)
txt(cx, 27.2, r"sharpness re-distributes over the", size=9)
txt(cx, 25.2, r"rank-$(n{-}1)$ orthogonal complement", size=9)
txt(cx, 21.8, r"only a \textbf{global} action escapes:", size=9)
txt(cx, 19.6, r"SAM drives $\lambda_{\max}$ below $2/\eta$", size=9,
    color=GREEN, weight="bold")
txt(cx, 16.2, r"\textit{scale-up agrees: ResNet-18/CIFAR-100}", size=8,
    color=GRAY)
txt(cx, 14.2, r"\textit{(SGD $\approx$ rank-1 $<$ SAM, clean CI;}", size=8,
    color=GRAY)
txt(cx, 12.2, r"\textit{sharp-escape arm: $-18.5\%$ test MSE)},", size=8,
    color=GRAY)
txt(cx, 9.0, r"\textbf{predictor} $\neq$ \textbf{actuator}", size=10)

# ------------------------------------------------------------- flow arrows
arrow(24.9, 26.0, 27.1, 26.0)                      # 1 -> 2
arrow(49.4, 33.0, 52.1, 36.8, color=BLUE)          # 2 -> 3
arrow(49.4, 20.0, 52.1, 16.2, color=RED)           # 2 -> 4
arrow(75.4, 35.5, 77.6, 33.0, color=BLUE)          # 3 -> 5
arrow(75.4, 14.5, 77.6, 17.0, color=RED)           # 4 -> 5

# ------------------------------------------------------------ takeaway line
txt(50, 3.4,
    r"\textit{takeaway: one HVP gives a reliable EoS monitor "
    r"($r \approx 0.98$); rank-$k$ local edits never push below $2/\eta$; "
    r"only the global action SAM flat-escapes}",
    size=9, color="#333333")

# ------------------------------------------------------------------ save
fig.savefig("figures/00_project_overview.pdf", pad_inches=0.02)
fig.savefig("figures/00_project_overview.png", dpi=300, pad_inches=0.02)
print("saved figures/00_project_overview.{pdf,png}")
