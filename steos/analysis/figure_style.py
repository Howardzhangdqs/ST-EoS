r"""R30 project-wide visual system for publication figures.

This module is the single source of truth for the ST-EoS figure identity.
Every plot script must ``setup()`` first and pull colors / hierarchy
constants from here instead of hardcoding its own.

RED LINES (do not change without an explicit decision):
  * vector PDF output — ``savefig`` writes ``figures/{name}.pdf``;
  * ``text.usetex=True`` with the Times text / Computer Modern math stack
    (matches PRIMEarxiv.sty); the font family is frozen;
  * ``pdf.fonttype`` / ``ps.fonttype`` = 42 (embed editable TrueType);
  * ``savefig()`` keeps its tight-crop default (rcParams ``savefig.bbox`` =
    "tight" + pad 0.02 in). Frozen-footprint pages use ``savefig_fixed``
    instead, which locally disables the crop.

Public interface (backward compatible): setup / savefig / METHOD_COLORS /
METHOD_LINESTYLES / METHOD_MARKERS / THRESH_COLOR / FIG_DIR.  New in R30:
C, THEORY, ST, FROZEN_PAGES, frozen_figsize, savefig_fixed, panel_label,
TEXTWIDTH_PT, INCLUDE_FRAC, report.

Usage:
    from steos.analysis.figure_style import (
        setup, savefig, METHOD_COLORS, ST, panel_label,
        frozen_figsize, savefig_fixed,
    )
    setup()                          # canvas already at final physical size
    # or, for an oversized source canvas that will be shrunk at inclusion:
    # setup(scale=2.33)              # see report() for the exact factor

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for ax, m in zip(axes, ("sgd", "sam")):
        ax.plot(x, y, color=METHOD_COLORS[m], lw=ST["primary"])
        panel_label(ax, "a", "Short title")   # (a) tag + 3-6 word title
    savefig(fig, "42_example")               # -> figures/42_example.pdf

All axis/title/legend strings that contain math MUST use LaTeX math mode
($...$), never unicode literals (no λ̂ / ‖g‖ / Δθ — use $\hat\lambda$ /
$\|\mathbf{g}\|$ / $\Delta\theta$).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; PDF is vector so DPI is irrelevant for the file
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")

# ---------------------------------------------------------------------------
# R30 semantic palette (categorical). Paul Tol muted hues, one meaning each.
#
# THREE IRON RULES (aesthetic_diagnosis_r29.md §4A):
#   1. Same method -> same hue, in every figure, no exceptions.
#   2. Variants of one family get NO new color: distinguish them with
#      solid/dashed linestyle and filled/open markers (redundant channel,
#      grayscale/colorblind safe).
#   3. Continuous scalar encodings (survival fraction, phase diagrams,
#      heatmaps) keep using cividis. NEVER paint continuous quantities with
#      this categorical palette.
C = {
    "neutral":      "#5B6470",  # baseline / SGD — neutral slate
    "damping":      "#4477AA",  # rank-1 damping / attenuation — muted blue
    "amplification": "#CC6677", # amplification — muted rose
    "sam":          "#44AA99",  # SAM / perturbed-point family — muted teal
    "linsam":       "#AA4499",  # linSAM / linearized correction — muted purple
    "aux":          "#999933",  # auxiliary controls (W-cap, γ-cap) — muted olive
    "theory":       "#3F444A",  # theory/reference — dark gray, never a method
    "theory_light": "#90969E",  # secondary theory / grid-like refs
}
THEORY = {"dark": C["theory"], "light": C["theory_light"]}

# Canonical method -> color registry (project-wide convention).
# RULE: the same method/quantity gets the SAME color in every paper figure.
#   SGD / baselines                     -> neutral slate
#   damping family (rank-1 clamped / plant, rank-k precond, clamp-free
#     variants, generic damping)         -> damping blue
#   amplification family                 -> amplification rose
#   SAM                                  -> sam teal
#   linSAM (+bn-matched variant)         -> linsam purple
#   auxiliary controls (W-cap, γ-cap)    -> aux olive
# Family VARIANTS (e.g. rank1 vs rank1_noclamp, linsam vs linsam_bnmatched)
# share the family hue and are separated by linestyle + filled/open marker —
# do NOT mint new colors for them. Threshold/reference lines (2/η etc.) use
# THRESH_COLOR / THEORY, never a method hue.
METHOD_COLORS = {
    # baseline
    "sgd": C["neutral"],
    # SAM / perturbed-point family
    "sam": C["sam"],
    # damping family (hue: damping blue)
    "rank1": C["damping"],
    "rank1_clamped": C["damping"],
    "xsam_clamped": C["damping"],
    "eos_precond": C["damping"],
    "rankk": C["damping"],
    "rankkx_d1": C["damping"],
    "damping": C["damping"],
    # damping-family variants (clamp-free / faithful): same hue, separated
    # by dashdot linestyle + open square markers at the call site
    "rank1_noclamp": C["damping"],
    "xsam_faithful": C["damping"],
    # amplification family (hue: amplification rose)
    "amplification": C["amplification"],
    "rank1_amp": C["amplification"],
    # linSAM family
    "linsam": C["linsam"],
    "linsam_bnmatched": C["linsam"],
    # auxiliary controls
    "wcap": C["aux"],
    "gcap": C["aux"],
    "gamma_cap": C["aux"],
    "aux": C["aux"],
}
THRESH_COLOR = C["theory"]  # theory/threshold reference lines — dark gray

# ---------------------------------------------------------------------------
# Per-method linestyle & marker registry — the redundant NON-COLOR coding
# channel (colorblind/grayscale friendly; aesthetic_diagnosis_r29.md §5 red
# line: never rely on hue alone). Same method = same linestyle/marker in
# every figure. Within a family, variants share the family entry and are
# told apart at the call site via filled vs open (markerfacecolor="none")
# markers and solid/dashed strokes:
#   sgd / baselines               -> solid    ●  (o)
#   damping family (rank-1/rank-k plant) -> dashed ▲ (^)
#   clamp-free/faithful variants  -> dashdot  ■  (s, drawn open)
#   amplification family          -> dashdot  ▼  (v)
#   SAM                           -> dotted   ◆  (D)
#   linSAM family                 -> longdash ✛  (P)
#   auxiliary caps (W/γ)          -> longdash ▼  (v)
METHOD_LINESTYLES = {
    "sgd": "-",
    "sam": ":",
    "rank1": "--",
    "rank1_clamped": "--",
    "xsam_clamped": "--",
    "eos_precond": "--",
    "rankk": "--",
    "rankkx_d1": "--",
    "damping": "--",
    "rank1_noclamp": "-.",
    "xsam_faithful": "-.",
    "amplification": "-.",
    "rank1_amp": "-.",
    "linsam": (0, (5, 2)),
    "linsam_bnmatched": (0, (5, 2)),
    "wcap": (0, (5, 2)),
    "gcap": (0, (5, 2)),
    "gamma_cap": (0, (5, 2)),
    "aux": (0, (5, 2)),
}
METHOD_MARKERS = {
    "sgd": "o",
    "sam": "D",
    "rank1": "^",
    "rank1_clamped": "^",
    "xsam_clamped": "^",
    "eos_precond": "^",
    "rankk": "^",
    "rankkx_d1": "^",
    "damping": "^",
    "rank1_noclamp": "s",
    "xsam_faithful": "s",
    "amplification": "v",
    "rank1_amp": "v",
    "linsam": "P",
    "linsam_bnmatched": "P",
    "wcap": "v",
    "gcap": "v",
    "gamma_cap": "v",
    "aux": "v",
}

# ---------------------------------------------------------------------------
# Visual-hierarchy constants (pt at FINAL physical size). setup(scale=s)
# refreshes every pt-valued entry by x s so scripts on oversized source
# canvases keep the same final stroke weights. alpha_* are dimensionless
# opacities and are deliberately scale-INVARIANT (alpha 0.10-0.15 is the
# final-size target range; scaling alpha would break the hierarchy).
#
# Usage (aesthetic_diagnosis_r29.md §4A "线条、CI、marker 的层级"):
#   ST["primary"]    primary mean curve lw 1.4-1.6 pt
#   ST["secondary"]  secondary method curve lw 1.1-1.3 pt
#   ST["theory"]     theory/threshold lines lw 0.8-0.9 pt, neutral gray
#   ST["seed"]       per-seed trajectories lw 0.5-0.6 pt (texture, not
#                    subject) with alpha=ST["alpha_seed"] (0.10-0.15)
#   ST["alpha_ci"]   CI bands: same hue as the mean, alpha 0.10-0.12
#   ST["marker"]     discrete sweep markers ~4 pt
#   ST["errbar"]     error bar lines 0.9 pt, caps ST["capsize"] ~2 pt
#   ST["panel"]      panel short-title fontsize; ST["tag"] panel tag size
ST = {
    "primary": 1.5,
    "secondary": 1.2,
    "theory": 0.85,
    "seed": 0.55,
    "errbar": 0.9,
    "alpha_seed": 0.12,
    "alpha_ci": 0.11,
    "marker": 4.0,
    "capsize": 2.0,
    "panel": 8.5,
    "tag": 8.5,
    "note": 8.0,
}

# ---------------------------------------------------------------------------
# Frozen page footprints (pt, from the r29 pdfinfo baseline). Figures whose
# layout footprint is frozen (main-text float flow must not move) MUST use
# figsize = frozen_figsize(name) + savefig_fixed(); all restyling happens
# INSIDE the original canvas.
FROZEN_PAGES = {
    "47_etad_collapse": (766.872, 324.0),  # R38: height 300.092 -> 324 for
    # the bottom 2-row legend (user: width 0.95\linewidth; panels ~11%
    # shorter than the 360 probe, legend band unchanged at bottom=0.30)
    "43_bsharp_arms": (535.975, 249.735),
    "46_tau_interpolation": (230.4, 309.6),
}

# \textwidth of the manuscript in pt (ICLR two-column 10pt: 396.0 pt) and
# the per-figure inclusion width as a fraction of it (from main.tex
# \includegraphics widths). Used by report() to close the sizing loop.
TEXTWIDTH_PT = 396.0
INCLUDE_FRAC = {
    "47_etad_collapse": 0.95,  # R38: 0.83 -> 0.95 (user, bottom-legend restyle)
    "43_bsharp_arms": 0.95,
    "46_tau_interpolation": 0.30,
    "fig1_rayleigh_monitor_ci": 0.92,
    "31_c4_lagged_controls": 0.95,
    "32_bridge_clampfree_c3": 0.92,
    "48_fixedd_phase": 0.90,
    "33_rank1_amplify": 0.75,
    "29_multisharp_migration": 0.95,
    "40_batch_sharpness": 0.95,
    "36_spectral_redistribution": 0.95,
    "45_regime_sweep": 1.00,
    "39_long_recipe_200ep": 0.95,
    "30_cifar100_r18_control": 0.95,
    "26_multiseed_control_results": 0.62,
    "27_rankk_sweep": 0.62,
    "41_linsam": 0.62,
    "44_rho_sweep": 0.95,
    "35_rankkx_interpolation": 0.62,
    "28_xsam_ablation": 0.95,
    "34_precond_sharpness": 0.95,
}

__all__ = [
    "FIG_DIR", "C", "THEORY", "METHOD_COLORS", "METHOD_LINESTYLES",
    "METHOD_MARKERS", "THRESH_COLOR", "ST", "FROZEN_PAGES", "TEXTWIDTH_PT",
    "INCLUDE_FRAC", "setup", "panel_label", "savefig", "savefig_fixed",
    "frozen_figsize", "report",
]


def setup(scale: float = 1.0, **overrides):
    r"""Set R30 rcParams for vector PDF + true LaTeX rendering.

    Call once at script top. Fonts match the manuscript exactly
    (PRIMEarxiv.sty): Times text (ptm) + Computer Modern math. Requires a
    TeX install (latex on PATH).

    ``scale`` is the source-canvas -> final-physical-size magnification:
    if the figure is drawn oversized and shrunk at \includegraphics time,
    pass scale = 1/s where s is the inclusion magnification (report()
    prints it). Every font size / linewidth / marker size / tick length /
    capsize below is multiplied by ``scale`` so the FINAL on-paper values
    hit the R30 baseline; figsize is never scaled here (choose canvas
    sizes per figure, frozen pages via frozen_figsize). ``overrides``
    (raw rcParams keys) are applied AFTER scaling, to tweak individual
    keys.

    Baseline (scale=1, i.e. final physical size): font 8.0 / title 8.5 /
    label 8.0 / tick 8.0 / legend 8.0 pt (R31: unified at one size just
    below the 10pt body text, per user directive; hierarchy is carried by
    weight/color, not size); axes lw 0.7; top+right spines
    off; grid OFF; lines 1.35 pt, markers 4 pt; ticks 2.5 x 0.7;
    frameless legends; errorbar capsize 2.0; patch lw 0.6.
    """
    s = float(scale)
    rc = {
        # --- vector PDF / font red lines (unchanged) ---
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,     # embed TrueType (editable in Illustrator)
        "ps.fonttype": 42,
        # True LaTeX rendering — same font packages as the paper
        # (PRIMEarxiv.sty sets \rmdefault{ptm}: Times text + CM math).
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{amsmath}"
            r"\usepackage{times}"
        ),
        # mathtext fallback (only used if usetex is disabled per-script)
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        # readable serif defaults (rm = Times via ptm under usetex)
        "font.family": "serif",
        "figure.dpi": 100,      # screen preview only; PDF is vector
        "savefig.dpi": 300,     # only matters if any raster (images) get embedded
        # --- R31 typography: one unified size, just below the 10pt body ---
        "font.size": 8.0 * s,
        "axes.titlesize": 8.5 * s,
        "axes.labelsize": 8.0 * s,
        "xtick.labelsize": 8.0 * s,
        "ytick.labelsize": 8.0 * s,
        "legend.fontsize": 8.0 * s,
        # --- R30 axes: fewer frames, no grid ---
        "axes.linewidth": 0.7 * s,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        # --- R30 strokes / markers ---
        "lines.linewidth": 1.35 * s,
        "lines.markersize": 4.0 * s,
        "lines.solid_capstyle": "round",
        "xtick.major.size": 2.5 * s,
        "ytick.major.size": 2.5 * s,
        "xtick.major.width": 0.7 * s,
        "ytick.major.width": 0.7 * s,
        # legend spacing keys are in fontsize units -> auto-scale, keep raw
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "errorbar.capsize": 2.0 * s,
        "patch.linewidth": 0.6 * s,
    }
    rc.update(overrides)
    plt.rcParams.update(rc)
    # refresh hierarchy constants (pt values scale; alphas do not — see ST)
    ST.update({
        "primary": 1.5 * s,
        "secondary": 1.2 * s,
        "theory": 0.85 * s,
        "seed": 0.55 * s,
        "errbar": 0.9 * s,
        "alpha_seed": 0.12,
        "alpha_ci": 0.11,
        "marker": 4.0 * s,
        "capsize": 2.0 * s,
        "panel": 8.5 * s,
        "tag": 8.5 * s,
        "note": 8.0 * s,
    })
    return rc


def panel_label(ax, tag: str, title: str = ""):
    """Place the R31 panel tag "(a)" + optional short title, CENTERED.

    Project-wide convention (must stay identical across all figures):
      * the whole "(a) Short title" string is centered above the panel
        (axes-fraction x=0.5), just above the axes' top edge, outside the
        data area;
      * fontsize ST["tag"] / ST["panel"] (8.5 pt at scale 1); the tag is
        bold (via usetex \\textbf), the title regular, both in #1A1D21 —
        hierarchy comes from weight, not size or position;
      * implemented as a single centered set_title so the visual center of
        the combined string sits exactly on the panel center.
    Titles are SHORT (3-6 words, e.g. "Fixed reference"); protocol and
    interpretation belong in the caption, not on the panel. title=""
    draws the tag only.
    """
    if title:
        full = rf"\textbf{{({tag})}} {title}"
    else:
        full = rf"\textbf{{({tag})}}"
    return ax.set_title(
        full, fontsize=ST["tag"], color="#1A1D21", pad=0.35 * ST["tag"],
    )


def frozen_figsize(name: str):
    """Return the frozen (width_in, height_in) for a frozen-footprint figure.

    Page footprint in pt is FROZEN_PAGES[name]; matplotlib figsize is in
    inches, hence pt/72. Pair with savefig_fixed() so the emitted MediaBox
    is exactly the r29 baseline.
    """
    w_pt, h_pt = FROZEN_PAGES[name]
    return (w_pt / 72.0, h_pt / 72.0)


def savefig(fig, name: str, ext: str = "pdf"):
    """Save figure to figures/{name}.{ext} (default PDF) and close it.

    bbox="tight" (rcParams) plus pad_inches=0.02 trims the white border to a
    hairline so figures sit flush in the paper (default pad 0.1in looks like a
    thick frame at \\linewidth scale). Frozen-footprint figures must use
    savefig_fixed() instead."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.{ext}"
    fig.savefig(out, pad_inches=0.02)
    plt.close(fig)
    return out


def savefig_fixed(fig, name: str, ext: str = "pdf"):
    """Save with the page EXACTLY figsize x 72 pt — no tight crop.

    For frozen-footprint figures (FROZEN_PAGES) the output MediaBox must
    stay bit-identical to the r29 baseline, so the tight-bbox rcParam is
    disabled for this save only (an explicit bbox_inches=None would fall
    back to the rcParam, hence the rc_context). Requirements:
      * use figsize=frozen_figsize(name) when creating the figure;
      * with no crop, anything near the canvas edge CLIPS — keep your own
        margins (subplots_adjust / constrained_layout padding ~0.35 in);
        panel tags/labels must not touch the border.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.{ext}"
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
    plt.close(fig)
    return out


def report(pdf_path, name: str):
    """QA the inclusion math of an emitted figure PDF (needs pdfinfo).

    Parses the page size via ``pdfinfo`` and prints:
      * page size (pt) and inclusion width INCLUDE_FRAC[name]*396 pt;
      * magnification s = frac * TEXTWIDTH_PT / page_width (source pt ->
        final on-paper pt);
      * recommended ``setup(scale=1/s)``;
      * the resulting final on-paper font table (base 8.0 / 8.5 / 8.0 /
        8.0 / 8.0 pt for font/title/label/tick/legend) and the floor
        checks tick & legend >= 8.0 pt, label >= 8.0 pt;
      * if name is in FROZEN_PAGES, MATCH/MISMATCH of the page size
        against the frozen r29 baseline (two-decimal tolerance).
    Returns the parsed numbers as a dict.
    """
    proc = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    )
    m = re.search(r"Page size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts", proc.stdout)
    if m is None:
        raise RuntimeError(f"could not parse pdfinfo page size for {pdf_path}")
    w_pt, h_pt = float(m.group(1)), float(m.group(2))
    frac = INCLUDE_FRAC[name]
    s = frac * TEXTWIDTH_PT / w_pt
    scale = 1.0 / s
    final = {
        "font": round(8.0 * scale * s, 2),
        "title": round(8.5 * scale * s, 2),
        "label": round(8.0 * scale * s, 2),
        "tick": round(8.0 * scale * s, 2),
        "legend": round(8.0 * scale * s, 2),
    }
    floor_tick = final["tick"] >= 8.0 and final["legend"] >= 8.0
    floor_label = final["label"] >= 8.0

    print(f"[report] {pdf_path}  ({name})")
    print(f"  page size       : {w_pt:.2f} x {h_pt:.2f} pt")
    print(f"  include width   : {frac:.2f} x {TEXTWIDTH_PT:.1f} = "
          f"{frac * TEXTWIDTH_PT:.2f} pt")
    print(f"  magnification s : {s:.4f}  (source pt -> final pt)")
    print(f"  setup scale     : {scale:.4f}  (= 1/s)")
    print(f"  final fonts     : font {final['font']:.2f}, title "
          f"{final['title']:.2f}, label {final['label']:.2f}, tick "
          f"{final['tick']:.2f}, legend {final['legend']:.2f}")
    print(f"  floors          : tick/legend >= 8.0 "
          f"{'PASS' if floor_tick else 'FAIL'} ; label >= 8.0 "
          f"{'PASS' if floor_label else 'FAIL'}")
    if name in FROZEN_PAGES:
        fw, fh = FROZEN_PAGES[name]
        match = round(w_pt, 2) == round(fw, 2) and round(h_pt, 2) == round(fh, 2)
        print(f"  frozen page     : {'MATCH' if match else 'MISMATCH'} "
              f"(baseline {fw:.2f} x {fh:.2f}, got {w_pt:.2f} x {h_pt:.2f})")
    else:
        print("  frozen page     : n/a (not a frozen figure)")
    return {
        "page_pt": (w_pt, h_pt), "frac": frac, "s": s, "scale": scale,
        "final_fonts": final, "floor_tick_legend": floor_tick,
        "floor_label": floor_label,
    }
