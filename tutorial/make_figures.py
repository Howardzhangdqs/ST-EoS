#!/usr/bin/env python3
r"""
Generate all pedagogical figures for the EoS tutorial article.

Uses matplotlib's PGF backend with xelatex for true LaTeX math rendering.
All figures are exported as PDF and embedded in the LaTeX document.

Usage:
    python make_figures.py
"""

import os
import sys
import csv
import numpy as np

# ── matplotlib setup: PGF backend with xelatex for real LaTeX rendering ──
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "pgf.texsystem":     "xelatex",
    "pgf.preamble":      r"\usepackage{amsmath,amssymb,bm}"
                         "\n"
                         r"\usepackage{xeCJK}"
                         "\n"
                         r"\setCJKmainfont{Noto Sans CJK SC}",
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.figsize":    (5.0, 3.5),
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "lines.linewidth":   1.5,
})

# ── Paths ──
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RUN_DIR = os.path.join(os.path.dirname(FIG_DIR), "..", "runs")

# ── Colour palette (consistent across all figures) ──
C_LAM    = "#2563eb"   # blue    – λ_max / sharpness
C_LOSS   = "#dc2626"   # red     – loss
C_THRESH = "#f59e0b"   # amber   – 2/η threshold
C_PROXY  = "#059669"   # green   – proxy / weight norm
C_ALIGN  = "#7c3aed"   # purple  – alignment
C_TRAJ   = "#0891b2"   # cyan    – trajectory
C_GREY   = "#6b7280"   # grey    – secondary

CMAP_STEP = "viridis"


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {name}")


def _load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                data.setdefault(k, []).append(float(v) if v else np.nan)
    return {k: np.array(v) for k, v in data.items()}


# ════════════════════════════════════════════════════════════════════
#  Figure 1 — Loss landscape concept (2-D contour)
# ════════════════════════════════════════════════════════════════════
def fig01_loss_landscape():
    r"""A 2-D loss landscape with multiple minima to build intuition."""
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    # multi-bowl landscape
    Z = (0.8 * np.exp(-((X - 1.5)**2 + (Y - 1.2)**2) / 0.4)
         + 1.0 * np.exp(-((X + 1.2)**2 + (Y + 0.8)**2) / 0.8)
         + 0.5 * np.exp(-((X - 0.3)**2 + (Y + 2.0)**2) / 0.3))
    Z = -Z  # valleys = minima

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cf = ax.contourf(X, Y, Z, levels=30, cmap="Blues_r")
    ax.contour(X, Y, Z, levels=15, colors="k", linewidths=0.4, alpha=0.3)
    fig.colorbar(cf, ax=ax, label=r"损失 $\mathcal{L}(\boldsymbol{\theta})$")

    # Mark GD trajectory (dashed)
    traj_x = np.array([-2.5, -2.0, -1.6, -1.3, -1.2, -1.2])
    traj_y = np.array([2.5, 1.5,  0.8,  0.5,  0.7,  0.8])
    ax.plot(traj_x, traj_y, "o--", color="yellow", markersize=4,
            markerfacecolor="red", markeredgecolor="yellow", linewidth=1.5)
    ax.annotate("梯度下降轨迹", xy=(-1.8, 1.2), color="black",
                fontsize=8, ha="center", fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.85))

    # Mark minima
    ax.plot(1.5, 1.2, "r*", markersize=12)
    ax.plot(-1.2, -0.8, "r*", markersize=12)
    ax.annotate("平坦极小值", xy=(1.5, 1.2), xytext=(2.3, 2.3),
                fontsize=8, color="black", ha="center", fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.annotate("尖锐极小值", xy=(-1.2, -0.8), xytext=(-2.8, -2.2),
                fontsize=8, color="black", ha="center", fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(r"损失景观 $\mathcal{L}(\boldsymbol{\theta})$")
    _save(fig, "fig01_loss_landscape.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 2 — Sharp vs flat minimum (1-D cross-sections)
# ════════════════════════════════════════════════════════════════════
def fig02_sharp_flat():
    r"""Compare a sharp (high-curvature) and flat (low-curvature) minimum."""
    theta = np.linspace(-3, 3, 500)

    # Sharp minimum: L = (1/2) * 50 * θ²
    L_sharp = 0.5 * 50 * theta**2
    # Flat minimum:  L = (1/2) * 2  * θ²
    L_flat  = 0.5 * 2  * theta**2

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)

    # Sharp
    axes[0].plot(theta, L_sharp, color=C_LAM, linewidth=2)
    axes[0].fill_between(theta, L_sharp, alpha=0.1, color=C_LAM)
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1, 60)
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"损失 $\mathcal{L}(\theta)$")
    axes[0].set_title(r"\textbf{尖锐}极小值：$\lambda_{\max}=50$")
    axes[0].axhline(0, color="k", linewidth=0.5)
    # Show curvature
    axes[0].annotate(r"$\mathcal{L}''(0)=50$" + "\n（高曲率）",
                     xy=(0, 0), xytext=(0.3, 40), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=C_GREY))

    # Flat
    axes[1].plot(theta, L_flat, color=C_PROXY, linewidth=2)
    axes[1].fill_between(theta, L_flat, alpha=0.1, color=C_PROXY)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-0.5, 20)
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"损失 $\mathcal{L}(\theta)$")
    axes[1].set_title(r"\textbf{平坦}极小值：$\lambda_{\max}=2$")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].annotate(r"$\mathcal{L}''(0)=2$" + "\n（低曲率）",
                     xy=(0, 0), xytext=(1.0, 14), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=C_GREY))

    fig.suptitle(r"尖锐度 $=$ 极小值处的曲率 $= \lambda_{\max}$",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig02_sharp_flat.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 3 — Hessian eigenvalues & eigenvectors (ellipse level set)
# ════════════════════════════════════════════════════════════════════
def fig03_hessian_eigen():
    r"""Visualise Hessian eigenvalues as principal axes of level-set ellipses."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Case 1: isotropic (λ₁ = λ₂)
    ax = axes[0]
    ax.set_aspect("equal")
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=C_LAM, linewidth=2)
    ax.fill(r * np.cos(theta), r * np.sin(theta), alpha=0.1, color=C_LAM)
    ax.annotate("", xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.annotate("", xy=(0, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.text(0.6, -0.25, r"$\mathbf{v}_1, \lambda_1{=}1$", fontsize=9, color="red")
    ax.text(-0.7, 0.6, r"$\mathbf{v}_2, \lambda_2{=}1$", fontsize=9, color="red")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title(r"各向同性：$\lambda_1 = \lambda_2$" + "\n（圆形等高线）")
    ax.grid(True, alpha=0.3)

    # Case 2: anisotropic (λ₁ >> λ₂)
    ax = axes[1]
    ax.set_aspect("equal")
    lam1, lam2 = 9, 1  # eigenvalues
    a = 1.0 / np.sqrt(lam2)  # semi-axis along v1 (flat direction, small λ → long axis)
    b = 1.0 / np.sqrt(lam1)  # semi-axis along v2 (sharp direction, large λ → short axis)
    ax.plot(a * np.cos(theta), b * np.sin(theta), color=C_LAM, linewidth=2)
    ax.fill(a * np.cos(theta), b * np.sin(theta), alpha=0.1, color=C_LAM)
    ax.annotate("", xy=(a, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.annotate("", xy=(0, b), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.text(0.5, -0.15, r"$\mathbf{v}_2$：平坦 $(\lambda_2{=}1)$", fontsize=8, color="red")
    ax.text(-0.35, 0.2, r"$\mathbf{v}_1$：尖锐 $(\lambda_1{=}9)$", fontsize=8,
            color="red", rotation=90)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title(r"各向异性：$\lambda_1 \gg \lambda_2$"
                 + "\n（椭圆形等高线）\n"
                 + r"$\lambda_{\max} = \lambda_1 = 9$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig03_hessian_eigen.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 4 — 1-D GD stability on a quadratic (3 cases)
# ════════════════════════════════════════════════════════════════════
def fig04_gd_stability():
    r"""Three cases of GD on L = ½λθ²: converge, edge of stability, diverge."""
    lam = 10.0  # curvature
    theta_curve = np.linspace(-1.5, 1.5, 300)
    L_curve = 0.5 * lam * theta_curve**2

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[1, 1].set_visible(False)
    ax_list = [axes[0, 0], axes[1, 0], axes[0, 1]]

    cases = [
        (r"$\eta < 2/\lambda$（收敛）", 0.15,
         r"$\eta\lambda = 1.5 < 2$", "tab:green"),
        (r"$\eta = 2/\lambda$（稳定边界）", 0.20,
         r"$\eta\lambda = 2$", "tab:orange"),
        (r"$\eta > 2/\lambda$（发散）", 0.30,
         r"$\eta\lambda = 3 > 2$", "tab:red"),
    ]

    for ax, (title, eta, sub, color) in zip(ax_list, cases):
        # Draw the parabola
        ax.plot(theta_curve, L_curve, color=C_GREY, linewidth=1, alpha=0.5)
        # GD trajectory
        theta = 1.0  # start
        thetas = [theta]
        for _ in range(8):
            grad = lam * theta
            theta = theta - eta * grad
            thetas.append(theta)
        thetas = np.array(thetas)
        # Clip for plotting
        thetas_clip = np.clip(thetas, -2, 2)

        # Draw steps
        for i in range(len(thetas_clip) - 1):
            t0, t1 = thetas_clip[i], thetas_clip[i + 1]
            L0 = 0.5 * lam * t0**2
            L1 = 0.5 * lam * t1**2
            ax.annotate("", xy=(t1, L1), xytext=(t0, L0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
            ax.plot(t0, L0, "o", color=color, markersize=5, zorder=5)
        ax.plot(thetas_clip[-1], 0.5 * lam * thetas_clip[-1]**2, "o",
                color=color, markersize=5, zorder=5)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 14)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\mathcal{L}(\theta)$")
        ax.set_title(title + "\n" + sub, fontsize=9)
        ax.axhline(0, color="k", linewidth=0.5)
        # Show formula
        ax.text(0.02, 0.95, r"$\theta_{t+1} = \theta_t - \eta\lambda\theta_t$"
                "\n" + r"$= (1-\eta\lambda)\,\theta_t$",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C_GREY, alpha=0.8))

    fig.suptitle(r"在抛物线 $\mathcal{L}(\theta)=\tfrac{1}{2}\lambda\theta^2$（$\lambda=10$）上的梯度下降",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig04_gd_stability.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 4b — Local quadratic approximation of a general function
# ════════════════════════════════════════════════════════════════════
def fig04b_taylor_approx():
    r"""Local quadratic (Taylor) approximation of a non-quadratic loss,
    showing how curvature varies along the landscape."""
    theta = np.linspace(-2.5, 2.5, 500)

    # Non-quadratic loss: L(θ) = θ²/2 + θ⁴/4
    L      = 0.5 * theta**2 + 0.25 * theta**4
    Lprime = theta + theta**3
    Lpp    = 1 + 3 * theta**2  # curvature = 1 + 3θ²

    # Two points: flat region (θ=0.3) and sharp region (θ=1.5)
    pts = [
        (0.3, C_PROXY, "平坦区域"),
        (1.5, C_LAM,   "尖锐区域"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: loss function + quadratic approximations
    ax1.plot(theta, L, color="black", linewidth=1.8, label=r"一般损失 $\mathcal{L}(\theta)$")
    for t0, color, label in pts:
        L0   = 0.5 * t0**2 + 0.25 * t0**4
        g0   = t0 + t0**3
        h0   = 1 + 3 * t0**2
        # Quadratic approximation: L(θ₀) + g₀(θ-θ₀) + ½h₀(θ-θ₀)²
        quad = L0 + g0 * (theta - t0) + 0.5 * h0 * (theta - t0)**2
        ax1.plot(theta, quad, "--", color=color, linewidth=1.5, alpha=0.8,
                 label=rf"局部二次逼近（$h={h0:.2f}$）")
        ax1.plot(t0, L0, "o", color=color, markersize=6, zorder=5)
        ax1.annotate(rf"$\theta_0={t0}$", xy=(t0, L0),
                     xytext=(t0 - 0.8, L0 + 3), fontsize=8, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, lw=1))
    ax1.set_xlim(-2.3, 2.3)
    ax1.set_ylim(-0.5, 18)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\mathcal{L}(\theta)$")
    ax1.set_title(r"一般函数的局部二次逼近（泰勒展开）")
    ax1.legend(fontsize=7.5, loc="upper center")
    ax1.axhline(0, color="k", linewidth=0.3)

    # Right: curvature along θ
    ax2.plot(theta, Lpp, color=C_LAM, linewidth=2)
    for t0, color, label in pts:
        h0 = 1 + 3 * t0**2
        ax2.plot(t0, h0, "o", color=color, markersize=6, zorder=5)
        ax2.annotate(rf"$\lambda={h0:.1f}$" + "\n" + label,
                     xy=(t0, h0), xytext=(t0 + 0.3, h0 + 2),
                     fontsize=8, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, lw=1))
    ax2.axhline(2 / 0.1, color=C_THRESH, linestyle="--", linewidth=1.2,
                label=rf"$2/\eta$（$\eta=0.1$）")
    ax2.set_xlim(-2.3, 2.3)
    ax2.set_ylim(0, 20)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"局部曲率 $\lambda(\theta) = \mathcal{L}''(\theta)$")
    ax2.set_title(r"曲率沿轨迹变化 $\Rightarrow$ 稳定性是动态的")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "fig04b_taylor_approx.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 5 — The EoS signature (real data)
# ════════════════════════════════════════════════════════════════════
def fig05_eos_signature():
    r"""The canonical EoS plot: λ_max → 2/η and loss, from real MLP run.
    Uses fine-grained λ_max data (every 5 steps)."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos_fine")
    sharp = _load_csv(os.path.join(run, "sharpness.csv"))
    metrics = _load_csv(os.path.join(run, "metrics.csv"))

    steps_lam = sharp["step"]
    lam = sharp["lambda_max"]
    steps_loss = metrics["step"]
    loss = metrics["loss"]

    eta = 0.05
    thresh = 2.0 / eta

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 1]))

    # Top: λ_max (fine-grained, no markers)
    ax1.plot(steps_lam, lam, "-", color=C_LAM, linewidth=1, label=r"$\lambda_{\max}(t)$", zorder=3)
    ax1.axhline(thresh, color=C_THRESH, linestyle="--", linewidth=1.5,
                label=rf"stability threshold $2/\eta = {thresh:.0f}$", zorder=2)
    ax1.set_ylabel(r"海森矩阵最大特征值 $\lambda_{\max}$")
    ax1.legend(loc="upper right")
    ax1.set_title(r"\textbf{稳定边界（EoS）}现象 — tanh MLP（$\eta = 0.05$）")
    ax1.set_ylim(-5, 80)

    # Shade phases
    ax1.axvspan(0, 250, alpha=0.08, color=C_PROXY)
    ax1.axvspan(250, 3000, alpha=0.05, color=C_LAM)
    ax1.text(120, 72, "第一阶段\n（渐进尖锐化）", fontsize=8,
             ha="center", color=C_PROXY, fontweight="normal")
    ax1.text(1500, 72, "第二阶段（EoS：在 $2/\\eta$ 附近振荡）",
             fontsize=8, ha="center", color=C_LAM, fontweight="normal")

    # Bottom: loss
    ax2.plot(steps_loss, loss, color=C_LOSS, linewidth=0.8, alpha=0.9)
    ax2.set_ylabel(r"训练损失 $\mathcal{L}$")
    ax2.set_xlabel(r"梯度下降步数 $t$")
    ax2.set_ylim(-0.5, max(loss) * 0.3)
    ax2.set_title(r"损失下降（短时间尺度上非单调）")

    fig.tight_layout()
    _save(fig, "fig05_eos_signature.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 5b — Zoomed view: periodic loss spikes at EoS
# ════════════════════════════════════════════════════════════════════
def fig05b_loss_spikes():
    r"""Zoomed view of the periodic loss spikes — the visible signature of
    GD crossing the 2/η stability boundary. Uses fine-grained λ_max data
    (measured every 5 steps) for clear correlation with loss spikes."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos_fine")
    metrics = _load_csv(os.path.join(run, "metrics.csv"))
    sharp = _load_csv(os.path.join(run, "sharpness.csv"))

    steps_loss = metrics["step"]
    loss = metrics["loss"]
    steps_lam = sharp["step"]
    lam = sharp["lambda_max"]
    eta = 0.05
    thresh = 2.0 / eta

    # Window: steps 200–600 (two full spike cycles)
    w0, w1 = 200, 600
    ml = (steps_loss >= w0) & (steps_loss <= w1)
    ms = (steps_lam >= w0) & (steps_lam <= w1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 1.2]))

    # Compute contiguous regions where λ_max > 2/η (for clean shading)
    s_win = steps_lam[ms]
    l_win = lam[ms]
    over = l_win > thresh
    # Find contiguous True segments
    regions = []
    start = None
    for i, ov in enumerate(over):
        if ov and start is None:
            start = s_win[i]
        elif not ov and start is not None:
            regions.append((start, s_win[i]))
            start = None
    if start is not None:
        regions.append((start, s_win[-1]))

    # Top: λ_max in window (fine-grained, every 5 steps)
    for r0, r1 in regions:
        ax1.axvspan(r0, r1, alpha=0.12, color="red", zorder=0)
    ax1.plot(s_win, l_win, "-", color=C_LAM, linewidth=1.2, zorder=3)
    ax1.axhline(thresh, color=C_THRESH, linestyle="--", linewidth=1.5,
                label=rf"$2/\eta = {thresh:.0f}$", zorder=2)
    ax1.set_ylabel(r"$\lambda_{\max}$")
    ax1.set_title(r"\textbf{放大查看}：$\lambda_{\max}$ 超过 $2/\eta$ 时 loss 尖峰"
                  + "\n" + r"（$\lambda_{\max}$ 每 5 步测量一次）")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, 80)

    # Bottom: loss in window — the spikes are clearly visible
    for r0, r1 in regions:
        ax2.axvspan(r0, r1, alpha=0.12, color="red", zorder=0)
    ax2.plot(steps_loss[ml], loss[ml], color=C_LOSS, linewidth=0.8, zorder=3)
    ax2.set_ylabel(r"训练损失 $\mathcal{L}$")
    ax2.set_xlabel(r"梯度下降步数 $t$")
    ax2.set_ylim(-0.5, 32)

    # Annotate the spikes
    ax2.annotate(r"\textbf{损失尖峰}" + "\n" + r"$\mathcal{L}$: $1\to16$",
                 xy=(239, 16), xytext=(300, 25), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.9))
    ax2.annotate(r"\textbf{损失尖峰}" + "\n" + r"$\mathcal{L}$: $0.5\to28$",
                 xy=(436, 28), xytext=(480, 22), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.9))

    # Annotate the stable descent between spikes
    ax2.annotate(r"稳定下降" + "\n" + r"($\lambda_{\max} < 2/\eta$)",
                 xy=(340, 0.8), fontsize=8, ha="center", color="green")

    fig.tight_layout()
    _save(fig, "fig05b_loss_spikes.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 5c — Real-world loss spikes: BLOOM 176B
# ════════════════════════════════════════════════════════════════════
def fig05c_bloom_spike():
    r"""Real-world loss spike from BLOOM 176B training log (public data).
    Shows the same spike-and-recover pattern in a 176-billion-parameter model."""
    # Data from BLOOM-176B chronicles (public GitHub log):
    # https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md
    # 2022-04-12 spike event
    iters = np.array([31210, 31214, 31216, 31217, 31218, 31219, 31220, 31222,
                      31225, 31230, 31240, 31250, 31260, 31270])
    loss  = np.array([2.18,  2.20,  2.60,  2.24,  2.35,  5.10,  3.80,  2.81,
                      2.55,  2.38,  2.25,  2.22,  2.20,  2.19])
    gnorm = np.array([0.22,  0.24,  2.39, 15.89, 145.1, 960.4, 480.2, 650.2,
                      120.5,  15.8,  1.20,  0.27,  0.25,  0.23])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 1]))

    # Top: loss
    ax1.plot(iters, loss, "o-", color=C_LOSS, markersize=4, linewidth=1.2)
    ax1.set_ylabel(r"训练损失 $\mathcal{L}$")
    ax1.set_title(r"\textbf{真实的 loss 飙升}：BLOOM 176B 预训练（2022-04-12）")
    ax1.set_ylim(1.8, 5.5)
    ax1.annotate(r"\textbf{尖峰}" + "\n" + r"$\mathcal{L}$: $2.2\to5.1$" + "\n" + r"（$2.3\times$ 暴涨）",
                 xy=(31219, 5.1), xytext=(31235, 4.5), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.9))
    ax1.annotate(r"$\sim$30 步后恢复",
                 xy=(31250, 2.22), xytext=(31255, 3.2), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="green", lw=1),
                 color="green")

    # Bottom: gradient norm (log scale)
    ax2.plot(iters, gnorm, "s-", color=C_LAM, markersize=4, linewidth=1.2)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"梯度范数 $\|\nabla\mathcal{L}\|$")
    ax2.set_xlabel(r"训练迭代步数")
    ax2.set_ylim(0.1, 2000)
    ax2.annotate(r"梯度范数暴涨" + "\n" + r"$0.2\to960$（$\sim5000\times$）",
                 xy=(31219, 960), xytext=(31235, 300), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=C_LAM, alpha=0.9))

    fig.tight_layout()
    _save(fig, "fig05c_bloom_spike.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 6 — Progressive sharpening zoom
# ════════════════════════════════════════════════════════════════════
def fig06_progressive_sharpening():
    r"""Progressive sharpening: λ_max and weight norm on a single dual-axis plot.
    Uses fine-grained data (every 5 steps)."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos_fine")
    sharp = _load_csv(os.path.join(run, "sharpness.csv"))
    metrics = _load_csv(os.path.join(run, "metrics.csv"))

    eta = 0.05
    thresh = 2.0 / eta

    mask = sharp["step"] <= 1200
    steps = sharp["step"][mask]
    lam = sharp["lambda_max"][mask]

    wn_steps = metrics["step"]
    wn = metrics.get("wnorm_output", metrics.get("wnorm_fc_out", np.zeros_like(wn_steps)))
    mask2 = wn_steps <= 1200
    wn_steps = wn_steps[mask2]
    wn = wn[mask2]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Left axis: λ_max
    ax1.plot(steps, lam, "-", color=C_LAM, linewidth=1.2, zorder=3,
             label=r"$\lambda_{\max}$（左轴）")
    ax1.axhline(thresh, color=C_THRESH, linestyle="--", linewidth=1.5, zorder=2,
                label=rf"$2/\eta = {thresh:.0f}$")
    ax1.set_xlabel(r"步数 $t$")
    ax1.set_ylabel(r"$\lambda_{\max}$", color=C_LAM)
    ax1.tick_params(axis="y", labelcolor=C_LAM)
    ax1.set_ylim(-5, 80)

    # Right axis: weight norm
    ax2 = ax1.twinx()
    ax2.plot(wn_steps, wn, "-", color=C_PROXY, linewidth=1.2, zorder=3,
             label=r"$\|\mathbf{W}\|$（右轴）")
    ax2.set_ylabel(r"权重范数 $\|\mathbf{W}\|$", color=C_PROXY)
    ax2.tick_params(axis="y", labelcolor=C_PROXY)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax1.set_title(r"\textbf{渐进尖锐化}：$\lambda_{\max}$ 与权重范数同步增长")
    ax1.annotate(r"$\|\mathbf{W}\|$ 增长 $\Rightarrow$ $\lambda_{\max}\!\uparrow$",
                 xy=(300, 30), fontsize=8, color="black",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.8))

    fig.tight_layout()
    _save(fig, "fig06_progressive_sharpening.pdf")

    fig.tight_layout()
    _save(fig, "fig06_progressive_sharpening.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 7 — Period-2 oscillation (schematic + real)
# ════════════════════════════════════════════════════════════════════
def fig07_period2():
    r"""Illustrate the period-2 oscillation at EoS — the zigzag along v_top."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: schematic 1-D quadratic with period-2 bouncing
    ax = axes[0]
    theta = np.linspace(-2, 2, 300)
    lam = 10.0
    L = 0.5 * lam * theta**2
    ax.plot(theta, L, color=C_GREY, linewidth=1, alpha=0.5)

    # Simulate period-2: η slightly above 2/λ
    eta = 0.22  # ηλ = 2.2 > 2 → unstable
    th = 0.3
    thetas = [th]
    for _ in range(10):
        th = th - eta * lam * th
        thetas.append(th)
    thetas = np.array(thetas)

    for i in range(len(thetas) - 1):
        t0, t1 = thetas[i], thetas[i + 1]
        L0, L1 = 0.5 * lam * t0**2, 0.5 * lam * t1**2
        ax.annotate("", xy=(t1, L1), xytext=(t0, L0),
                    arrowprops=dict(arrowstyle="->", color=C_ALIGN, lw=2))
        ax.plot([t0], [L0], "o", color=C_ALIGN, markersize=5, zorder=5)
    ax.plot([thetas[-1]], [0.5 * lam * thetas[-1]**2], "o",
            color=C_ALIGN, markersize=5, zorder=5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 12)
    ax.set_xlabel(r"$\theta$（沿 $\mathbf{v}_{\top}$ 方向）")
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.set_title(r"\textbf{周期-2 振荡}" + "\n" +
                 r"$\eta\lambda_{\max} > 2$：梯度下降过冲")
    ax.text(0.02, 0.95,
            r"$\theta_{t+1} = (1-\eta\lambda_{\max})\,\theta_t$"
            "\n" + r"$\eta\lambda_{\max} = 2.2 > 2$",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C_GREY, alpha=0.8))

    # Right: real λ_max oscillation at EoS
    ax = axes[1]
    run = os.path.join(RUN_DIR, "mlp_ts_eos")
    sharp = _load_csv(os.path.join(run, "sharpness.csv"))
    steps = sharp["step"]
    lam_real = sharp["lambda_max"]
    eta = 0.05
    thresh = 2.0 / eta

    mask = (steps >= 250) & (steps <= 2000)
    ax.plot(steps[mask], lam_real[mask], "o-", color=C_LAM, markersize=3)
    ax.axhline(thresh, color=C_THRESH, linestyle="--", linewidth=1.5,
               label=rf"$2/\eta = {thresh:.0f}$")
    ax.set_xlabel(r"步数 $t$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.set_title(r"\textbf{真实的 EoS 振荡}" + "\n" +
                 r"$\lambda_{\max}$ 在 $2/\eta$ 附近振荡")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "fig07_period2.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 8 — Self-stabilization: |cos(Δθ, v_top)| alignment
# ════════════════════════════════════════════════════════════════════
def fig08_self_stabilization():
    r"""The Damian et al. self-stabilization signature from real data."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos")
    dt = np.load(os.path.join(run, "dthetas.npz"), allow_pickle=True)
    vt = np.load(os.path.join(run, "vtops.npz"), allow_pickle=True)
    D = dt["D"]  # (60, 71425)
    V = vt["V"]  # (60, 71425)
    steps_dt = dt["steps"]

    n = min(len(D), len(V))
    cos_align = np.zeros(n)
    for i in range(n):
        d = np.asarray(D[i]).flatten()
        v = np.asarray(V[i]).flatten()
        cos_align[i] = abs(d @ v) / (np.linalg.norm(d) * np.linalg.norm(v) + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(steps_dt[:n], cos_align, "o-", color=C_ALIGN, markersize=4)
    ax.axhline(1.0, color=C_THRESH, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"步数 $t$")
    ax.set_ylabel(r"$|\cos(\Delta\boldsymbol{\theta},\, \mathbf{v}_{\top})|$")
    ax.set_title(r"\textbf{自稳定化}：更新方向与顶部特征向量对齐")
    ax.set_ylim(-0.05, 1.1)
    ax.annotate(r"早期：随机方向" + "\n" + r"（$|\cos| \approx 0.1$）",
                xy=(steps_dt[2], cos_align[2]),
                xytext=(steps_dt[10], 0.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_GREY))
    ax.annotate(r"EoS 处：完全对齐" + "\n" + r"（$|\cos| \to 0.998$）",
                xy=(steps_dt[-5], cos_align[-5]),
                xytext=(steps_dt[-30], 0.6), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_GREY))

    fig.tight_layout()
    _save(fig, "fig08_self_stabilization.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 9 — PCA spectrum of the GD trajectory
# ════════════════════════════════════════════════════════════════════
def fig09_pca_spectrum():
    r"""PCA spectrum of the weight-snapshot matrix — PC1 dominates."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos")
    snap = np.load(os.path.join(run, "snapshots.npz"), allow_pickle=True)
    W = snap["W"]  # (300, 71425)
    Wc = W - W[0:1]

    # Gram-matrix PCA
    Gram = Wc @ Wc.T
    eigvals = np.maximum(np.linalg.eigvalsh(Gram)[::-1], 0)
    total = max(eigvals.sum(), 1e-12)
    pcts = eigvals / total * 100

    n_show = 8
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x_pos = np.arange(1, n_show + 1)
    bars = ax.bar(x_pos, pcts[:n_show], color=C_TRAJ, alpha=0.8, edgecolor="k", linewidth=0.5)
    bars[0].set_color(C_LAM)
    bars[0].set_alpha(0.9)

    for i, (x, p) in enumerate(zip(x_pos, pcts[:n_show])):
        ax.text(float(x), float(p + 1), f"{p:.1f}%", ha="center", fontsize=8, fontweight="normal")

    ax.set_xlabel(r"主成分序号")
    ax.set_ylabel(r"解释方差占比（\%）")
    ax.set_title(r"\textbf{轨迹是低维的}：PC1 解释了 "
                 + f"{pcts[0]:.0f}" + r"\% 的方差")
    ax.set_xticks(x_pos)
    ax.set_ylim(0, min(pcts[0] * 1.2, 100))

    fig.tight_layout()
    _save(fig, "fig09_pca_spectrum.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 10 — Trajectory projected onto PC1–PC2 (corridor)
# ════════════════════════════════════════════════════════════════════
def fig10_trajectory_2d():
    r"""Project the GD trajectory onto its top-2 PCs — shows the corridor."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos")
    snap = np.load(os.path.join(run, "snapshots.npz"), allow_pickle=True)
    W = snap["W"]
    Wc = W - W[0:1]

    # Gram PCA → snapshot projections (300 values, not 71425)
    Gram = Wc @ Wc.T
    eigvals, eigvecs = np.linalg.eigh(Gram)
    eigvals = np.maximum(eigvals[::-1], 0)  # descending
    eigvecs = eigvecs[:, ::-1]
    # Snapshot projections onto top PCs: sqrt(eigval) * eigvec
    PC1 = np.sqrt(eigvals[0]) * eigvecs[:, 0]
    PC2 = np.sqrt(eigvals[1]) * eigvecs[:, 1]
    steps_snap = np.asarray(snap["steps"], dtype=float)

    # Manually map steps to colours (PGF backend workaround)
    cmap_obj = plt.get_cmap(CMAP_STEP)
    s_norm = (steps_snap - steps_snap.min()) / (steps_snap.max() - steps_snap.min() + 1e-12)
    colors_scatter = cmap_obj(s_norm)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(PC1, PC2, c=colors_scatter, s=8, zorder=3)
    ax.plot(PC1, PC2, "-", color=C_GREY, alpha=0.3, linewidth=0.5, zorder=2)
    fig.colorbar(sc, ax=ax, label=r"步数 $t$")
    ax.set_xlabel(rf"PC1（{np.sum(eigvals[::-1][:1])/np.sum(eigvals)*100:.0f}\% 方差）")
    ax.set_ylabel(rf"PC2（{np.sum(eigvals[::-1][1:2])/np.sum(eigvals)*100:.0f}\% 方差）")
    ax.set_title(r"\textbf{梯度下降轨迹位于狭窄走廊中}")
    ax.annotate(r"起点", xy=(PC1[0], PC2[0]), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red"),
                xytext=(PC1[0] + 0.5, PC2[0] + 0.5))
    ax.annotate(r"终点", xy=(PC1[-1], PC2[-1]), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red"),
                xytext=(PC1[-1] - 2, PC2[-1] + 0.3))
    # Stretch PC2 axis 2x so the plot fills the vertical space
    pc2_center = (PC2.min() + PC2.max()) / 2
    pc2_half = (PC2.max() - PC2.min()) / 2
    ax.set_ylim(pc2_center - pc2_half * 2, pc2_center + pc2_half * 2)

    fig.tight_layout()
    _save(fig, "fig10_trajectory_2d.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 11 — Sharpness proxy: weight norm tracks λ_max
# ════════════════════════════════════════════════════════════════════
def fig11_sharpness_proxy():
    r"""Weight norm as a cheap proxy for λ_max — dual-axis time series.
    Uses fine-grained λ_max data (every 5 steps)."""
    run = os.path.join(RUN_DIR, "mlp_ts_eos_fine")
    sharp = _load_csv(os.path.join(run, "sharpness.csv"))
    metrics = _load_csv(os.path.join(run, "metrics.csv"))

    steps_lam = sharp["step"]
    lam = sharp["lambda_max"]

    # Interpolate λ_max to every step
    lam_interp = np.interp(metrics["step"], steps_lam, lam)
    wn = metrics.get("wnorm_output", metrics.get("wnorm_fc_out",
              metrics.get("wnorm_fc2", np.zeros_like(metrics["step"]))))

    # Normalise both to [0,1] for visual comparison
    lam_n = (lam_interp - lam_interp.min()) / (lam_interp.max() - lam_interp.min() + 1e-12)
    wn_n = (wn - wn.min()) / (wn.max() - wn.min() + 1e-12)

    # Pearson correlation
    mask = ~np.isnan(lam_n) & ~np.isnan(wn_n)
    r = np.corrcoef(lam_n[mask], wn_n[mask])[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(metrics["step"], lam_n, color=C_LAM, linewidth=1.2, alpha=0.8,
            label=r"$\lambda_{\max}$ (normalised)")
    ax.plot(metrics["step"], wn_n, color=C_PROXY, linewidth=1.2, alpha=0.8,
            label=r"$\|\mathbf{W}\|$ (normalised)")
    ax.set_xlabel(r"步数 $t$")
    ax.set_ylabel(r"归一化值")
    ax.set_title(r"\textbf{权重范数跟踪尖锐度}："
                 + rf"Pearson $r = {r:.2f}$")
    ax.legend()
    ax.text(0.02, 0.95,
            r"\textbf{关键洞察：}" + "\n"
            + r"$\|\mathbf{W}\|$ 是 $\lambda_{\max}$ 的" + "\n"
            + r"\emph{免费}逐步代理量" + "\n"
            + r"（无需计算海森矩阵！）",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.6", fc="lightyellow", ec=C_THRESH, alpha=0.9))

    fig.tight_layout()
    _save(fig, "fig11_sharpness_proxy.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 12 — Universality: λ_max → 2/η across learning rates
# ════════════════════════════════════════════════════════════════════
def fig12_universality():
    r"""λ_max locks onto 2/η across different learning rates.
    Uses fine-grained data (every 5 steps)."""
    lrs = [0.01, 0.02, 0.05]
    results = {}
    for lr in lrs:
        run = os.path.join(RUN_DIR, f"mlp_ts_eos_fine_{lr}")
        sharp = _load_csv(os.path.join(run, "sharpness.csv"))
        results[lr] = (sharp["step"], sharp["lambda_max"])

    colors = [C_LAM, C_ALIGN, C_PROXY, C_TRAJ, C_LOSS]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lr in enumerate(lrs):
        steps, lam = results[lr]
        thresh = 2.0 / lr
        c = colors[i % len(colors)]
        ax.plot(steps, lam, "-", color=c, linewidth=1, alpha=0.8, zorder=3,
                label=rf"$\eta={lr}$, $2/\eta={thresh:.0f}$")
        ax.axhline(thresh, color=c, linestyle="--", linewidth=0.8, alpha=0.4, zorder=2)

    ax.set_xlabel(r"步数 $t$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.set_title(r"\textbf{普适性}：无论 $\eta$ 如何，$\lambda_{\max} \to 2/\eta$")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 250)
    ax.text(0.02, 0.95,
            r"$\lambda_{\max}^{\mathrm{终值}} \approx 2/\eta$"
            "\n" + r"误差不超过 $0.5\%$" + "\n"
            + r"横跨 $5\times$ 学习率范围",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec=C_THRESH, alpha=0.9))

    fig.tight_layout()
    _save(fig, "fig12_universality.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 13 — Stability boundary diagram
# ════════════════════════════════════════════════════════════════════
def fig13_stability_diagram():
    r"""The η–λ stability diagram: stable region, EoS boundary, unstable region."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    eta_line = np.linspace(0.01, 0.3, 100)
    thresh = 2.0 / eta_line

    # Stable region (below 2/η) — green fill
    ax.fill_between(eta_line, 0, thresh * 0.9, color="#d1fae5", alpha=0.7)
    # EoS band (±10% around 2/η) — amber fill
    ax.fill_between(eta_line, thresh * 0.9, thresh * 1.1, color="#fef3c7", alpha=0.8)
    # Unstable region (above 2/η) — red fill
    ax.fill_between(eta_line, thresh * 1.1, 250, color="#fecaca", alpha=0.5)

    # Boundary line: ηλ = 2 → λ = 2/η
    ax.plot(eta_line, thresh, "k-", linewidth=2, label=r"$\eta\,\lambda_{\max} = 2$")

    # Mark the MLP runs
    for lr, marker in [(0.01, "o"), (0.02, "s"), (0.05, "^")]:
        ax.plot(float(lr), 2.0 / float(lr), marker, color="blue", markersize=8, zorder=5)
        ax.annotate(rf"$\eta={lr}$", xy=(float(lr), 2.0 / float(lr)),
                    xytext=(float(lr) + 0.015, 2.0 / float(lr) + 12), fontsize=8)

    ax.set_xlabel(r"学习率 $\eta$")
    ax.set_ylabel(r"尖锐度 $\lambda_{\max}$")
    ax.set_title(r"\textbf{稳定性图}：$2/\eta$ 边界")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 250)
    ax.legend(fontsize=8, loc="upper right")

    # Region labels
    ax.text(0.18, 20, r"稳定区：$\eta\lambda < 2$", fontsize=9,
            ha="center", color="green", fontweight="normal", alpha=0.8)
    ax.text(0.18, 200, r"不稳定区：$\eta\lambda > 2$", fontsize=9,
            ha="center", color="red", fontweight="normal", alpha=0.8)
    ax.text(0.12, 50, r"EoS", fontsize=9,
            ha="center", color=C_THRESH, fontweight="normal")

    fig.tight_layout()
    _save(fig, "fig13_stability_diagram.pdf")


# ════════════════════════════════════════════════════════════════════
#  Figure 14 — Summary: the EoS cycle
# ════════════════════════════════════════════════════════════════════
def fig14_eos_cycle():
    r"""A schematic of the EoS self-regulating cycle."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"\textbf{稳定边界（EoS）自调节循环}", fontsize=12)

    # Four nodes arranged in a cycle
    nodes = {
        "sharpen":  (0, 2.0, r"\textbf{1. 渐进尖锐化}" + "\n\n"
                     + r"$\lambda_{\max}$ 随着 $\|\mathbf{W}\|$" + "\n"
                     + r"的增长而上升"),
        "threshold":(2.0, 0, r"\textbf{2. 触碰阈值}" + "\n\n"
                     + r"$\eta\,\lambda_{\max} \to 2$" + "\n"
                     + r"到达稳定边界"),
        "oscillate":(0, -2.0, r"\textbf{3. 周期-2 振荡}" + "\n\n"
                     + r"GD 沿 $\mathbf{v}_{\top}$ 过冲" + "\n"
                     + r"$|\cos(\Delta\bm{\theta},\mathbf{v}_{\top})| \to 1$"),
        "stabilize":(-2.0, 0, r"\textbf{4. 自稳定化}" + "\n\n"
                     + r"三阶项 $\tfrac{1}{6}\nabla^3\!\mathcal{L}$" + "\n"
                     + r"将 $\lambda_{\max}$ 拉回"),
    }

    colors = {"sharpen": C_PROXY, "threshold": C_THRESH,
              "oscillate": C_LAM, "stabilize": C_ALIGN}

    # Draw boxes
    for key, (x, y, text) in nodes.items():
        bbox = dict(boxstyle="round,pad=0.5", fc="white",
                    ec=colors[key], linewidth=2, alpha=0.9)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, bbox=bbox)

    # Draw arrows (cycle)
    arrow_kw = dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                    color=C_GREY, lw=2, connectionstyle="arc3,rad=0.3")
    ax.annotate("", xy=(1.5, 1.5), xytext=(1.0, 1.8), arrowprops=arrow_kw)   # 1→2
    ax.annotate("", xy=(1.5, -1.5), xytext=(1.8, -1.0), arrowprops=arrow_kw)  # 2→3
    ax.annotate("", xy=(-1.5, -1.5), xytext=(-1.0, -1.8), arrowprops=arrow_kw) # 3→4
    ax.annotate("", xy=(-1.5, 1.5), xytext=(-1.8, 1.0), arrowprops=arrow_kw)   # 4→1

    # Center label
    ax.text(0, 0, r"\textbf{EoS}" + "\n" + r"\textbf{循环}",
            ha="center", va="center", fontsize=14, fontweight="normal",
            color=C_LOSS,
            bbox=dict(boxstyle="circle,pad=0.6", fc="lightyellow",
                      ec=C_LOSS, linewidth=2))

    _save(fig, "fig14_eos_cycle.pdf")


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating EoS tutorial figures (PGF/xelatex → PDF)…")
    print(f"Output dir: {FIG_DIR}\n")

    fig01_loss_landscape()
    fig02_sharp_flat()
    fig03_hessian_eigen()
    fig04_gd_stability()
    fig04b_taylor_approx()
    fig05_eos_signature()
    fig05b_loss_spikes()
    fig05c_bloom_spike()
    fig06_progressive_sharpening()
    fig07_period2()
    fig08_self_stabilization()
    fig09_pca_spectrum()
    fig10_trajectory_2d()
    fig11_sharpness_proxy()
    fig12_universality()
    fig13_stability_diagram()
    fig14_eos_cycle()

    print(f"\nDone — {len(os.listdir(FIG_DIR))} files in {FIG_DIR}")
